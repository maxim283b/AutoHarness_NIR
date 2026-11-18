"""Versioned native context policy, shared by optimizer and all target harnesses.

No API payload rewriting. Native OpenCode compaction uses the session's model;
its requests retain the parent accounting role and are tagged separately.
"""
from pathlib import Path

VERSION = "native-compaction-v1"
INPUT_LIMIT = 32000
TOOL_OUTPUT_BYTES = 8000

PLUGIN = r'''import { mkdir, writeFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import path from "node:path";

export default async ({ directory }, options = {}) => ({
  "chat.headers": async (input, output) => {
    output.headers["x-autoharness-purpose"] = input.agent === "compaction" ? "compaction" : "primary";
    output.headers["x-autoharness-context-policy"] = "native-compaction-v1";
  },
  "chat.params": async (input, output) => {
    if (input.agent === "compaction") output.maxOutputTokens = 4096;
  },
  "tool.execute.after": async (input, output) => {
    if (typeof output.output !== "string") return;
    const data = Buffer.from(output.output, "utf8");
    if (data.length <= 8000) return;
    const hash = createHash("sha256").update(data).digest("hex");
    const folder = options.outputDirectory || path.join(directory, ".autosaddler", "context-outputs");
    await mkdir(folder, { recursive: true });
    const file = path.join(folder, hash + ".txt");
    await writeFile(file, data);
    output.output = data.subarray(0, 5600).toString("utf8") +
      "\n[Output excerpt: full " + data.length + " bytes saved at " + file +
      ". Read bounded line ranges or search this file for further details.]\n" +
      data.subarray(-1600).toString("utf8");
    output.metadata = { ...output.metadata, autoharness_full_output: file, autoharness_original_bytes: data.length };
  },
});
'''


def write_plugin(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(PLUGIN.encode("utf-8"))
    return path


def configure(config, model, plugin_url, output_directory=None):
    """Apply the same frozen overlay without changing the candidate's files."""
    config["compaction"] = {"auto": True, "prune": True, "reserved": 8192}
    config.setdefault("agent", {})["compaction"] = {"model": model}
    config["agent"]["title"] = {"disable": True}
    config["agent"]["summary"] = {"disable": True}
    config.setdefault("plugin", []).append([plugin_url, {"outputDirectory":output_directory}] if output_directory else plugin_url)
    provider, model_id = model.split("/", 1)
    config["provider"][provider]["models"][model_id]["limit"]["input"] = INPUT_LIMIT
    return config
