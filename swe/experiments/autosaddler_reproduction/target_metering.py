"""Accounting tags only; target context and execution remain candidate mutable."""
VERSION = 'target-metering-v6'
PLUGIN_URL = 'file:///opt/autoharness/target-metering.mjs'
PLUGIN = '''export default async () => ({
  "chat.headers": async (input, output) => {
    output.headers["x-autoharness-purpose"] = input.agent === "compaction" ? "compaction" : "primary";
    output.headers["x-autoharness-context-policy"] = "target-metering-v6";
  },
});
'''
