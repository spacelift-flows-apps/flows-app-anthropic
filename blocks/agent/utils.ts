import Anthropic from "@anthropic-ai/sdk";
import { events, kv, timers } from "@slflows/sdk/v1";

interface ToolDefinition {
  name: string;
  description: string;
  schema: Anthropic.Messages.Tool.InputSchema;
}

interface CallState {
  messages: Anthropic.Beta.Messages.BetaMessageParam[];
  toolCallIds: string[];
  pendingId: string;
  toolDefinitions: ToolDefinition[];
  force: boolean | string;
  maxTokens: number;
  model: string;
  systemPrompt: string | undefined;
  turn: number;
  maxRetries: number;
  schema: Anthropic.Messages.Tool.InputSchema | undefined;
  thinking: boolean | undefined;
  thinkingBudget: number | undefined;
  temperature: number | undefined;
}

function joinToolNames(
  toolCalls: Anthropic.Beta.Messages.BetaToolUseBlock[],
  toolNames: Record<string, string>,
) {
  if (toolCalls.length === 0) {
    return "";
  }

  if (toolCalls.length === 1) {
    return `"${toolNames[toolCalls[0].name]}"`;
  }

  return `${toolCalls
    .slice(0, -1)
    .map((toolCall) => `"${toolNames[toolCall.name]}"`)
    .join(", ")} and "${toolNames[toolCalls[toolCalls.length - 1].name]}"`;
}

function streamMessage(params: {
  apiKey: string;
  model: string;
  maxTokens: number;
  temperature?: number | undefined;
  messages: Anthropic.Beta.Messages.BetaMessageParam[];
  systemPrompt?: string | undefined;
  tools: Anthropic.Tool[];
  force: boolean | string;
  thinking?: boolean | undefined;
  thinkingBudget?: number | undefined;
  schema?: Anthropic.Messages.Tool.InputSchema | undefined;
}) {
  const {
    apiKey,
    maxTokens,
    temperature,
    systemPrompt,
    model,
    messages,
    tools,
    schema,
    force,
    thinking,
    thinkingBudget,
  } = params;

  const client = new Anthropic({
    apiKey,
  });

  const shouldCallSpecificTool = tools.length > 0 && typeof force === "string";
  const shouldCallAnyTool = tools.length > 0 && force === true;

  return client.beta.messages.stream({
    max_tokens: maxTokens,
    temperature,
    system: systemPrompt,
    model,
    messages,
    tools,
    thinking:
      thinking && thinkingBudget
        ? {
            type: "enabled",
            budget_tokens: thinkingBudget,
          }
        : undefined,
    tool_choice:
      tools.length > 0
        ? shouldCallSpecificTool
          ? {
              type: "tool",
              name: force as string,
            }
          : shouldCallAnyTool
            ? {
                type: "any",
              }
            : {
                type: "auto",
              }
        : undefined,
    output_format: schema
      ? {
          type: "json_schema",
          schema: {
            ...schema,
            additionalProperties: false,
          },
        }
      : undefined,
    betas: ["mcp-client-2025-04-04", "structured-outputs-2025-11-13"],
  });
}

export function validateConfig(
  appConfig: Record<string, any>,
  staticConfig: Record<string, any>,
  inputConfig: Record<string, any>,
) {
  if (!appConfig.anthropicApiKey) {
    throw new Error("Anthropic API key is required");
  }

  const model = staticConfig.model ?? appConfig.defaultModel;

  if (!model) {
    throw new Error("Model is required");
  }

  if (
    inputConfig.thinking?.enabled &&
    (!inputConfig.thinking?.budget ||
      inputConfig.thinking?.budget >= inputConfig.maxTokens)
  ) {
    throw new Error(
      "You need to set thinking budget to a value less than max tokens",
    );
  }

  return {
    model,
    apiKey: appConfig.anthropicApiKey as string,
    toolDefinitions: (staticConfig.toolDefinitions ?? []) as ToolDefinition[],
    prompt: inputConfig.prompt as string,
    maxTokens: inputConfig.maxTokens as number,
    systemPrompt: inputConfig.systemPrompt as string | undefined,
    force: inputConfig.force as boolean | string,
    thinking: inputConfig.thinking?.enabled as boolean | undefined,
    thinkingBudget: inputConfig.thinking?.budget as number | undefined,
    schema: staticConfig.outputSchema as
      | Anthropic.Messages.Tool.InputSchema
      | undefined,
    maxRetries: (inputConfig.maxRetries ?? 1) as number,
    temperature: inputConfig.temperature as number | undefined,
  };
}

// Anthropic allows names that match only the following regex:
// ^[a-zA-Z0-9_-]{1,64}$
// https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/implement-tool-use#specifying-client-tools
function cleanToolName(name: string) {
  return name
    .trim() // Trim whitespace from both ends
    .toLowerCase() // Convert to lowercase
    .replace(/\s+/g, "_") // Replace all whitespace runs with a single underscore
    .replace(/[^a-z0-9_-]/g, "_") // Replace any character that is NOT a letter, number, underscore, or hyphen with an underscore
    .replace(/[_-]{2,}/g, "_") // Collapse consecutive underscores or hyphens into a single underscore
    .replace(/^[_-]+|[_-]+$/g, "") // Remove leading/trailing underscores or hyphens
    .slice(0, 64); // Ensure the name does not exceed 64 characters
}

function processToolDefinitions(toolDefinitions: ToolDefinition[]) {
  const tools: Anthropic.Tool[] = [];
  const toolNames: Record<string, string> = {};
  const toolOutputKeys: Record<string, string> = {};

  for (const toolDefinition of toolDefinitions) {
    const cleanedName = cleanToolName(toolDefinition.name);

    tools.push({
      name: cleanedName,
      description: toolDefinition.description,
      input_schema: toolDefinition.schema || {
        type: "object",
      },
    });

    toolNames[cleanedName] = toolDefinition.name;

    toolOutputKeys[cleanedName] = getToolDefinitionOutputKey(
      toolDefinition.name,
    );
  }

  return { tools, toolNames, toolOutputKeys };
}

async function syncPendingEventWithStream(
  pendingId: string,
  stream: ReturnType<typeof streamMessage>,
) {
  let lastTool: { name: string; serverName: string; id: string } | undefined;

  for await (const event of stream) {
    if (event.type !== "content_block_start") {
      continue;
    }

    switch (event.content_block.type) {
      case "mcp_tool_use": {
        await events.updatePending(pendingId, {
          statusDescription: `Calling "${event.content_block.name}" on "${event.content_block.server_name}"`,
        });

        lastTool = {
          name: event.content_block.name,
          serverName: event.content_block.server_name,
          id: event.content_block.id,
        };
        break;
      }
      case "mcp_tool_result": {
        if (lastTool && lastTool.id === event.content_block.tool_use_id) {
          await events.updatePending(pendingId, {
            statusDescription: `Received result of "${lastTool.name}" from "${lastTool.serverName}"`,
          });
        }

        break;
      }
      case "text": {
        await events.updatePending(pendingId, {
          statusDescription: "Processing...",
        });

        break;
      }
      case "thinking": {
        await events.updatePending(pendingId, {
          statusDescription: "Thinking...",
        });

        break;
      }
    }
  }
}

async function emitResult(
  pendingId: string,
  result: {
    output: unknown;
    usage: {
      inputTokens: number;
      outputTokens: number;
    };
  },
  eventIds: string[],
): Promise<void> {
  await events.emit(
    {
      output: result.output,
      usage: result.usage,
    },
    {
      complete: pendingId,
      parentEventId: eventIds.length > 0 ? eventIds[0] : undefined,
      secondaryParentEventIds:
        eventIds.length > 1 ? eventIds.slice(1) : undefined,
    },
  );
}

async function storeCallState(params: {
  executionId: string;
  eventIds: string[];
  pendingId: string;
  messages: Anthropic.Beta.Messages.BetaMessageParam[];
  toolCalls: Anthropic.Beta.Messages.BetaToolUseBlock[];
  toolDefinitions: ToolDefinition[];
  force: boolean | string;
  maxTokens: number;
  model: string;
  systemPrompt: string | undefined;
  maxRetries: number;
  schema: Anthropic.Messages.Tool.InputSchema | undefined;
  turn: number;
  thinking: boolean | undefined;
  thinkingBudget: number | undefined;
  temperature: number | undefined;
}) {
  const { executionId, toolCalls, ...rest } = params;

  await kv.block.set({
    key: `call-${executionId}`,
    value: {
      toolCallIds: toolCalls.map((toolCall) => toolCall.id),
      ...rest,
    } satisfies CallState,
    ttl: 60 * 60,
  });
}

export async function loadCallState(executionId: string) {
  const { value } = await kv.block.get(`call-${executionId}`);

  if (!value) {
    throw new Error("Call state not found");
  }

  return value as CallState;
}

async function deleteCallState(executionId: string) {
  await kv.block.delete([`call-${executionId}`]);
}

export async function storeToolResult(params: {
  executionId: string;
  toolCallId: string;
  result: unknown;
  turn: number;
  eventId: string;
}) {
  const { executionId, toolCallId, result, turn, eventId } = params;

  await kv.block.set({
    key: `result-${executionId}-${turn}-${toolCallId}`,
    value: {
      toolCallId,
      result,
      eventId,
    },
    ttl: 60 * 5,
  });
}

export async function loadToolResults(params: {
  executionId: string;
  turn: number;
  toolCallIds: string[];
}) {
  const { executionId, turn, toolCallIds } = params;

  const results = await kv.block.list({
    keyPrefix: `result-${executionId}-${turn}-`,
  });

  const toolResults = results.pairs.reduce(
    (acc, { value }) => {
      acc[value.toolCallId] = value.result;
      return acc;
    },
    {} as Record<string, string>,
  );

  const haveAllResults = toolCallIds.every(
    (toolCallId) => typeof toolResults[toolCallId] !== "undefined",
  );

  return {
    haveAllResults,
    toolResults,
    eventIds: results.pairs.map(({ value }) => value.eventId),
  };
}

export async function setTimeoutTimer(executionId: string) {
  const id = await timers.block.set(60 * 2, {
    description: "Waiting for tool results",
    inputPayload: {
      executionId,
    },
  });

  await kv.block.set({
    key: `timer-${executionId}`,
    value: id,
  });
}

export async function clearTimeoutTimer(executionId: string) {
  const { value } = await kv.block.get(`timer-${executionId}`);

  if (value) {
    await timers.block.unset(value);
  }
}

export async function continueTurn(params: {
  executionId: string;
  eventIds: string[];
  pendingId: string;
  messages: Anthropic.Beta.Messages.BetaMessageParam[];
  toolCallIds: string[];
  toolDefinitions: ToolDefinition[];
  toolResults: Record<string, string>;
  force: boolean | string;
  model: string;
  maxTokens: number;
  systemPrompt: string | undefined;
  turn: number;
  maxRetries: number;
  schema: Anthropic.Messages.Tool.InputSchema | undefined;
  apiKey: string;
  thinking: boolean | undefined;
  thinkingBudget: number | undefined;
  temperature: number | undefined;
}): Promise<void> {
  const {
    executionId,
    eventIds,
    pendingId,
    messages,
    toolCallIds,
    toolDefinitions,
    toolResults,
    force,
    model,
    maxTokens,
    systemPrompt,
    turn,
    maxRetries,
    schema,
    apiKey,
    thinking,
    thinkingBudget,
    temperature,
  } = params;

  await events.updatePending(pendingId, {
    statusDescription: `Received results from tool${
      toolCallIds.length === 1 ? "" : "s"
    }...`,
  });

  const nextMessages: Anthropic.Beta.Messages.BetaMessageParam[] = [
    ...messages,
    {
      role: "user",
      content: toolCallIds
        .map((id) => {
          const result = toolResults[id];

          if (typeof result === "undefined") {
            return null;
          }

          return {
            type: "tool_result" as const,
            tool_use_id: id,
            content: result,
          };
        })
        .filter((part) => part !== null),
    },
  ];

  return executeTurn({
    executionId,
    pendingId,
    eventIds,
    messages: nextMessages,
    toolDefinitions,
    force,
    model,
    maxTokens,
    systemPrompt,
    turn,
    apiKey,
    maxRetries,
    schema,
    thinking,
    thinkingBudget,
    temperature,
  });
}

async function handleModelResponse(params: {
  message: Anthropic.Beta.Messages.BetaMessage;
  pendingId: string;
  eventIds: string[];
  executionId: string;
  previousMessages: Anthropic.Beta.Messages.BetaMessageParam[];
  toolDefinitions: ToolDefinition[];
  force: boolean | string;
  model: string;
  maxTokens: number;
  systemPrompt: string | undefined;
  turn: number;
  maxRetries: number;
  schema: Anthropic.Messages.Tool.InputSchema | undefined;
  thinking: boolean | undefined;
  thinkingBudget: number | undefined;
  temperature: number | undefined;
}): Promise<void> {
  const {
    message,
    pendingId,
    eventIds,
    executionId,
    previousMessages,
    toolDefinitions,
    force,
    model,
    maxTokens,
    systemPrompt,
    turn,
    maxRetries,
    schema,
    thinking,
    thinkingBudget,
    temperature,
  } = params;

  const { toolNames, toolOutputKeys } = processToolDefinitions(toolDefinitions);
  if (message.stop_reason === "end_turn") {
    const textPart = message.content.findLast(
      (content) => content.type === "text",
    );

    if (!textPart?.text) {
      throw new Error("Model did not respond with text");
    }

    let output = textPart.text;

    if (schema) {
      try {
        output = JSON.parse(textPart.text);
      } catch {
        console.error("Failed to parse structured output");
      }
    }

    try {
      return emitResult(
        pendingId,
        {
          output,
          usage: {
            inputTokens: message.usage.input_tokens,
            outputTokens: message.usage.output_tokens,
          },
        },
        eventIds,
      );
    } finally {
      await deleteCallState(executionId);
    }
  }

  if (message.stop_reason === "tool_use") {
    const toolCalls = message.content.filter(
      (content) => content.type === "tool_use",
    );

    const toolCallNames = joinToolNames(toolCalls, toolNames);

    await events.updatePending(pendingId, {
      statusDescription:
        toolCalls.length === 1
          ? `Calling tool: ${toolCallNames}`
          : `Calling tools: ${toolCallNames}`,
    });

    await Promise.all(
      toolCalls.map((toolCall) =>
        events.emit(
          {
            parameters: toolCall.input,
            toolCallId: toolCall.id,
            executionId,
          },
          {
            outputKey: toolOutputKeys[toolCall.name],
            echo: true,
            parentEventId: eventIds.length > 0 ? eventIds[0] : undefined,
            secondaryParentEventIds:
              eventIds.length > 1 ? eventIds.slice(1) : undefined,
          },
        ),
      ),
    );

    await storeCallState({
      executionId,
      eventIds,
      messages: previousMessages.concat({
        role: message.role,
        content: message.content,
      }),
      toolCalls,
      pendingId,
      toolDefinitions,
      force,
      model,
      maxTokens,
      systemPrompt,
      turn: turn + 1,
      maxRetries,
      schema,
      thinking,
      thinkingBudget,
      temperature,
    });

    return setTimeoutTimer(executionId);
  }

  await events.cancelPending(pendingId, "Unexpected response from model");
  await deleteCallState(executionId);
}

export async function executeTurn(params: {
  executionId: string;
  pendingId: string;
  eventIds: string[];
  messages: Anthropic.Beta.Messages.BetaMessageParam[];
  toolDefinitions: ToolDefinition[];
  force: boolean | string;
  model: string;
  maxTokens: number;
  systemPrompt: string | undefined;
  turn: number;
  apiKey: string;
  maxRetries: number;
  schema: Anthropic.Messages.Tool.InputSchema | undefined;
  thinking: boolean | undefined;
  thinkingBudget: number | undefined;
  temperature: number | undefined;
}): Promise<void> {
  const {
    executionId,
    pendingId,
    eventIds,
    messages,
    toolDefinitions,
    force,
    model,
    maxTokens,
    systemPrompt,
    turn,
    apiKey,
    maxRetries,
    schema,
    thinking,
    thinkingBudget,
    temperature,
  } = params;

  let retryCount = 0;
  let lastError: Error | undefined;

  while (retryCount < maxRetries) {
    try {
      if (retryCount > 0) {
        await events.updatePending(pendingId, {
          statusDescription: `Retrying API call... (attempt ${retryCount + 1})`,
        });
      }

      const { tools } = processToolDefinitions(toolDefinitions);

      const stream = streamMessage({
        maxTokens,
        systemPrompt,
        model,
        messages,
        tools,
        force,
        apiKey,
        thinking,
        thinkingBudget,
        temperature,
        schema,
      });

      await syncPendingEventWithStream(pendingId, stream);

      const message = await stream.finalMessage();

      return handleModelResponse({
        message,
        pendingId,
        eventIds,
        executionId,
        previousMessages: messages,
        toolDefinitions,
        force,
        model,
        maxTokens,
        systemPrompt,
        turn,
        maxRetries,
        schema,
        thinking,
        thinkingBudget,
        temperature,
      });
    } catch (error) {
      lastError = error instanceof Error ? error : new Error(String(error));
      retryCount++;

      // Check if this is a retryable error (overloaded, rate limit, etc.)
      const errorMessage = lastError.message.toLowerCase();
      const isRetryable =
        errorMessage.includes("overloaded") ||
        errorMessage.includes("rate limit") ||
        errorMessage.includes("timeout") ||
        errorMessage.includes("502") ||
        errorMessage.includes("503") ||
        errorMessage.includes("504");

      // If not retryable or this was the last retry, exit
      if (!isRetryable || retryCount >= maxRetries) {
        break;
      }

      // Wait a bit before retrying (exponential backoff)
      const delay = Math.min(1000 * Math.pow(2, retryCount - 1), 10000);
      await new Promise((resolve) => setTimeout(resolve, delay));
    }
  }

  // All retries failed
  await events.cancelPending(
    pendingId,
    `API call failed: ${lastError?.message || "Unknown error"}`,
  );
  await deleteCallState(executionId);
  throw lastError || new Error("Unknown error");
}

export async function isTimerLocked(executionId: string): Promise<boolean> {
  const { value } = await kv.block.get(`lock-${executionId}`);
  return Boolean(value);
}

export async function setTimerLock(executionId: string): Promise<void> {
  await kv.block.set({
    key: `lock-${executionId}`,
    value: true,
    ttl: 60 * 5,
  });
}

export async function clearTimerLock(executionId: string): Promise<void> {
  await kv.block.delete([`lock-${executionId}`]);
}

export function getToolDefinitionOutputKey(name: string) {
  return `tool_definition_${name.toLowerCase().replace(/\s+/g, "_")}`;
}
