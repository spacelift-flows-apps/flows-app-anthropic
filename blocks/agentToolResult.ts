import { AppBlock, messaging } from "@slflows/sdk/v1";
import { parseInvocationId } from "./agent/utils";

export const agentToolResult: AppBlock = {
  name: "Agent Tool Result",
  category: "Core",
  description:
    "Routes processed tool results back to the Agent block. Connect this after your tool processing pipeline to return results to the agent.",
  inputs: {
    default: {
      config: {
        result: {
          name: "Result",
          description:
            "The result of the tool call to send back to the agent. Use `previous` to send the result of the previous tool call.",
          type: "string",
          required: true,
        },
        invocationId: {
          name: "Invocation ID",
          description:
            "The invocation ID from the agent's tool call output. This is automatically populated when connected to an agent's tool output.",
          type: "string",
          required: true,
          populateFrom: {
            blockKey: "agent",
            outputProperty: "invocationId",
          },
        },
      },
      onEvent: async (input) => {
        const { result, invocationId } = input.event.inputConfig;

        const { blockId, executionId, toolCallId } =
          parseInvocationId(invocationId);

        await messaging.sendToBlocks({
          body: {
            executionId,
            toolCallId,
            result,
            eventId: input.event.id,
          },
          blockIds: [blockId],
        });
      },
    },
  },
};
