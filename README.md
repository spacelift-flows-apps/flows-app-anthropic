# Anthropic

## Description

## Config

The app supports two authentication backends, auto-detected from which credentials you fill in:

- **Anthropic API** — set `Anthropic API Key`. Use standard Anthropic model IDs (e.g. `claude-sonnet-4-6`).
- **AWS Bedrock** — set `AWS Access Key ID`, `AWS Secret Access Key`, `AWS Region` (and optionally `AWS Session Token` for temporary STS credentials). Use Bedrock cross-region inference profile IDs, e.g. `global.anthropic.claude-sonnet-4-6` (or a regional variant like `us.anthropic.claude-sonnet-4-6`).

If both are provided, the Anthropic key takes precedence.

For auto-refreshing short-lived AWS credentials, pair this app with the **AWS STS** app and reference its credential signals in the AWS fields, so you don't need to manage long-lived IAM keys.

## Blocks

- `generateMessage`
  - Description: Generates a message based on the provided prompt and parameters. Supports schema-based object generation, tool calls, retry logic, "thinking" options, and integration with remote MCP servers.

- `toolDefinition`
  - Description: Defines a custom tool that can be invoked by the Generate message block. Exposes the tool definition via a signal and provides an input to process and return the tool's result.

- `remoteMcpServer`
  - Description: Configures a remote MCP server that Claude can leverage while generating messages. Validates connectivity, lists available tools, and exposes its configuration via a signal.
