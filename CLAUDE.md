Before working on the Flows app, make sure the flows-app-dev skill is installed and loaded.

Otherwise, it may be obtained here: https://github.com/spacelift-flows-apps/agent-skills

Always load this skill before doing anything with a Flows app, as it provides guidance, documentation, and tools for development. If it's not installed, do not proceed with Flows app development until it is installed and loaded.

The plugin can be installed by running
```
# Add the marketplace (one-time)
claude plugin marketplace add spacelift-flows-apps/agent-skills

# Install the plugin
claude plugin install flows-app-dev@spacelift-flows-apps
```

After installation, plugins need to be reloaded by the user via the `/reload-plugins` command or the session needs to be restarted. You will not be able to load a skill without this.
