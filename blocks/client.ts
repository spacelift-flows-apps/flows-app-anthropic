import Anthropic from "@anthropic-ai/sdk";
import AnthropicBedrock from "@anthropic-ai/bedrock-sdk";

export const ANTHROPIC_DEFAULT_MODEL = "claude-sonnet-4-6";
export const BEDROCK_DEFAULT_MODEL = "global.anthropic.claude-sonnet-4-6";

export type AnthropicAuth =
  | { kind: "anthropic"; apiKey: string }
  | {
      kind: "bedrock";
      awsAccessKey: string;
      awsSecretKey: string;
      awsRegion: string;
      awsSessionToken?: string;
    };

export function resolveAuth(appConfig: Record<string, any>): AnthropicAuth {
  if (appConfig.anthropicApiKey) {
    return { kind: "anthropic", apiKey: appConfig.anthropicApiKey };
  }

  if (appConfig.awsAccessKeyId && appConfig.awsSecretAccessKey) {
    return {
      kind: "bedrock",
      awsAccessKey: appConfig.awsAccessKeyId,
      awsSecretKey: appConfig.awsSecretAccessKey,
      awsRegion: appConfig.awsRegion,
      awsSessionToken: appConfig.awsSessionToken || undefined,
    };
  }

  throw new Error(
    "Provide either an Anthropic API key, or AWS access key + secret for the Bedrock backend.",
  );
}

export function defaultModelFor(auth: AnthropicAuth): string {
  return auth.kind === "bedrock"
    ? BEDROCK_DEFAULT_MODEL
    : ANTHROPIC_DEFAULT_MODEL;
}

export function createClient(
  auth: AnthropicAuth,
): Anthropic | AnthropicBedrock {
  if (auth.kind === "anthropic") {
    return new Anthropic({ apiKey: auth.apiKey });
  }

  return new AnthropicBedrock({
    awsAccessKey: auth.awsAccessKey,
    awsSecretKey: auth.awsSecretKey,
    awsRegion: auth.awsRegion,
    awsSessionToken: auth.awsSessionToken,
  });
}
