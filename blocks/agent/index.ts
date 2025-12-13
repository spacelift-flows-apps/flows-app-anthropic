import { AppBlock, AppBlockComponentOutput, events } from "@slflows/sdk/v1";

import {
  executeTurn,
  loadCallState,
  storeToolResult,
  loadToolResults,
  setTimeoutTimer,
  clearTimeoutTimer,
  continueTurn,
  isTimerLocked,
  setTimerLock,
  clearTimerLock,
  validateConfig,
  getToolDefinitionOutputKey,
} from "./utils";
import { randomUUID } from "node:crypto";

export const agent: AppBlock = {
  name: "Agent",
  category: "Core",
  description: "An agent that can call tools and generate structured output.",
  config: {
    model: {
      name: "Model",
      type: "string",
      description:
        "The model to use for the agent. Defaults to the default model set in the app config.",
      required: false,
    },
    toolDefinitions: {
      name: "Tools",
      description:
        "Array of tool blocks to use.\nYou can specify them like this:\n```javascript\n[signals.toolBlock.definition, ...]\n```",
      type: {
        type: "array",
        items: {
          type: "object",
          properties: {
            name: {
              title: "Name",
              type: "string",
              description: "Name of the tool",
            },
            description: {
              title: "Description",
              type: "string",
              description: "Description of what the tool does",
            },
            schema: {
              title: "Schema",
              type: "jsonSchemaObject",
              description: "Schema of the tool's input",
            },
          },
          required: ["name", "description"],
        },
      },
      required: false,
    },
    outputSchema: {
      name: "Output schema",
      description:
        "Used to constrain the output of the model to follow a specific schema, ensuring valid, parseable output for downstream processing.",
      type: {
        type: "jsonSchemaObject",
      },
      required: false,
    },
  },
  inputs: {
    default: {
      config: {
        prompt: {
          name: "Prompt",
          description: "The input to the model.",
          type: "string",
          required: true,
        },
        systemPrompt: {
          name: "System prompt",
          description:
            "A system prompt is a way of providing context and instructions to Claude, such as specifying a particular goal or role.",
          type: "string",
          required: false,
        },
        maxTokens: {
          name: "Max tokens",
          description:
            "The maximum number of tokens to generate before stopping. Note that the models may stop before reaching this maximum. This parameter only specifies the absolute maximum number of tokens to generate.",
          type: "number",
          default: 4096,
          required: false,
        },
        maxRetries: {
          name: "Max retries",
          description:
            "The number of times to retry the call if it fails to generate a valid object. Works only if schema is provided.",
          type: "number",
          required: false,
        },
        thinking: {
          name: "Thinking",
          description:
            "Whether to enable Claude's extended thinking. This will make the model think more deeply and generate more detailed responses. This will also increase the cost of the request.",
          type: {
            type: "object",
            properties: {
              enabled: {
                title: "Enabled",
                type: "boolean",
                description: "Whether to enable thinking",
              },
              budget: {
                title: "Budget",
                type: "number",
                description: "The number of tokens to use for thinking",
              },
            },
            required: ["enabled", "budget"],
          },
          required: false,
          default: {
            enabled: true,
            budget: 2048,
          },
        },
        force: {
          name: "Force",
          description:
            "Force the model to call a tool. Provide a name to call a specific tool or `true` to always call any tool.",
          type: {
            anyOf: [
              {
                type: "string",
                description: "The name of the tool to call.",
              },
              { type: "boolean", description: "Always call any tool." },
            ],
          },
          required: false,
          default: false,
        },
        temperature: {
          name: "Temperature",
          description:
            "Amount of randomness injected into the response. Defaults to `1.0`. Ranges from `0.0` to `1.0`. Use temperature closer to `0.0` for analytical / multiple choice, and closer to `1.0` for creative and generative tasks. Note that even with temperature of `0.0`, the results will not be fully deterministic.",
          type: "number",
          required: false,
        },
      },
      onEvent: async (input) => {
        const {
          toolDefinitions,
          prompt,
          model,
          maxTokens,
          systemPrompt,
          force,
          thinking,
          thinkingBudget,
          apiKey,
          schema,
          maxRetries,
          temperature,
        } = validateConfig(
          input.app.config,
          input.block.config,
          input.event.inputConfig,
        );

        const pendingId = await events.createPending({
          statusDescription: "Calling Anthropic model...",
        });

        const executionId = randomUUID();

        return executeTurn({
          executionId,
          pendingId,
          eventIds: [input.event.id],
          messages: [
            {
              role: "user",
              content: prompt,
            },
          ],
          toolDefinitions,
          force,
          model,
          maxTokens,
          systemPrompt,
          turn: 0,
          apiKey,
          maxRetries,
          schema,
          thinking,
          thinkingBudget,
          temperature,
        });
      },
    },
    toolResult: {
      name: "Tool result",
      config: {
        result: {
          name: "Result",
          description: "The result of the tool call",
          type: "any",
          required: true,
        },
      },
      onEvent: async (input) => {
        if (!input.event.echo) {
          throw new Error("This input should not be called directly");
        }

        const { body } = input.event.echo;
        const executionId = body.executionId;
        const { result } = input.event.inputConfig;
        const eventId = input.event.id;

        if (await isTimerLocked(executionId)) {
          return;
        }

        const {
          messages,
          toolCallIds,
          pendingId,
          toolDefinitions,
          force,
          model,
          turn,
          maxTokens,
          systemPrompt,
          maxRetries,
          schema,
          thinking,
          thinkingBudget,
          temperature,
        } = await loadCallState(executionId);

        // Clear the timeout in case we get all results and can either continue or complete.
        await clearTimeoutTimer(executionId);

        // Store the results separately to avoid race conditions.
        await storeToolResult({
          executionId,
          toolCallId: body.toolCallId,
          result: typeof result === "string" ? result : JSON.stringify(result),
          turn,
          eventId,
        });

        // Load the results to check if we have all of them.
        const { haveAllResults, toolResults, eventIds } = await loadToolResults(
          {
            executionId,
            turn,
            toolCallIds,
          },
        );

        if (!haveAllResults) {
          // If we don't have all results yet, we set a timeout to check again in 2 minutes.
          // This is to avoid waiting forever or race conditions.
          return setTimeoutTimer(executionId);
        }

        return continueTurn({
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
          thinking,
          thinkingBudget,
          temperature,
          apiKey: input.app.config.anthropicApiKey,
        });
      },
    },
  },

  onTimer: async (input) => {
    const { executionId } = input.timer.payload;

    await setTimerLock(executionId);

    try {
      const {
        messages,
        toolCallIds,
        pendingId,
        toolDefinitions,
        force,
        model,
        turn,
        maxTokens,
        systemPrompt,
        maxRetries,
        schema,
        thinking,
        thinkingBudget,
        temperature,
      } = await loadCallState(executionId);

      const { haveAllResults, toolResults, eventIds } = await loadToolResults({
        executionId,
        turn,
        toolCallIds,
      });

      if (!haveAllResults) {
        // Still waiting – cancel pending event and exit.
        return events.cancelPending(pendingId, "Timeout");
      }

      await continueTurn({
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
        thinking,
        thinkingBudget,
        temperature,
        apiKey: input.app.config.anthropicApiKey,
      });
    } finally {
      await clearTimerLock(executionId);
    }
  },

  onSync: async (input) => {
    const outputUpdates: Record<
      string,
      Partial<AppBlockComponentOutput> | null
    > = {};

    for (const outputKey of Object.keys(input.block.outputs)) {
      const output = input.block.outputs[outputKey];

      if (output && output.modified) {
        outputUpdates[outputKey] = null;
      }
    }

    outputUpdates["result"] = {
      name: "Result",
      default: true,
      possiblePrimaryParents: ["default"],
      type: {
        type: "object",
        properties: {
          output: input.block.config.outputSchema ?? {
            type: "string",
            description: "The generated message",
          },
          usage: {
            type: "object",
            properties: {
              inputTokens: { type: "number" },
              outputTokens: { type: "number" },
            },
          },
        },
        required: ["output", "usage"],
      },
    };

    let order = 10;

    for (const toolDefinition of input.block.config.toolDefinitions ?? []) {
      outputUpdates[getToolDefinitionOutputKey(toolDefinition.name)] = {
        name: toolDefinition.name,
        description: toolDefinition.description,
        type: {
          type: "object",
          properties: {
            parameters: toolDefinition.schema,
          },
          required: ["parameters"],
        },
        possiblePrimaryParents: ["toolResult"],
        order: order++,
        secondary: true,
      };
    }

    return {
      newStatus: "ready",
      outputUpdates:
        Object.keys(outputUpdates).length > 0 ? outputUpdates : undefined,
    };
  },
};
