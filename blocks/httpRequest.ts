import { AppBlock, events } from "@slflows/sdk/v1";
import { resolveAuth, createClient } from "./client";

export const httpRequest: AppBlock = {
  name: "HTTP Request",
  description:
    "Make a direct request to the Anthropic API. Use this as an escape hatch for any API endpoint not covered by dedicated blocks.",
  category: "Request",
  inputs: {
    default: {
      config: {
        method: {
          name: "HTTP Method",
          description: "HTTP method for the request.",
          type: {
            enum: ["GET", "POST", "PUT", "PATCH", "DELETE"],
          },
          required: true,
        },
        path: {
          name: "Path",
          description:
            'The API path (e.g., "/v1/models" or "/v1/messages/batches").',
          type: "string",
          required: true,
        },
        queryParams: {
          name: "Query Parameters",
          description: "Optional query parameters as key-value pairs.",
          type: {
            type: "object",
            additionalProperties: true,
          },
          required: false,
        },
        headers: {
          name: "Headers",
          description: "Optional extra HTTP headers as key-value pairs.",
          type: {
            type: "object",
            additionalProperties: { type: "string" },
          },
          required: false,
        },
        requestBody: {
          name: "Body",
          description: "Optional request body as key-value pairs.",
          type: {
            type: "object",
            additionalProperties: true,
          },
          required: false,
        },
      },
      onEvent: async (input) => {
        const auth = resolveAuth(input.app.config);
        const method = (input.event.inputConfig.method as string).toLowerCase();
        const path = input.event.inputConfig.path as string;
        const bodyValue = input.event.inputConfig.requestBody as
          | Record<string, unknown>
          | undefined;
        const queryParamsValue = input.event.inputConfig.queryParams as
          | Record<string, string>
          | undefined;
        const headersValue = input.event.inputConfig.headers as
          | Record<string, string>
          | undefined;

        const client = createClient(auth);
        const hasBody = method !== "get" && method !== "delete";

        const data = await client.request({
          method: method as "get" | "post" | "put" | "patch" | "delete",
          path,
          body: hasBody ? bodyValue : undefined,
          query: queryParamsValue,
          headers: headersValue,
        });

        await events.emit(data);
      },
    },
  },
  outputs: {
    result: {
      name: "Result",
      description: "The API response.",
      default: true,
      possiblePrimaryParents: ["default"],
      type: {
        type: "object",
        additionalProperties: true,
      },
    },
  },
};
