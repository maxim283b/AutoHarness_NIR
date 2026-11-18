"""OpenCode-specific optimizer instructions; underlying typed schemas stay compatible."""
import hashlib
import json
from dataclasses import replace
from autosaddler.v2.core.domain import to_json_value
from autosaddler.v2.plugins.meta_are.prompt_pack import MetaAREPromptPack
from autosaddler.v2.core.adaptive_memory import AdaptiveReflectionMemory

VERSION='opencode-full-source-v6-evidence-receipts'

GUIDES={
 'history-analysis':'Read .autosaddler/session_context.json and the history manifest. Inspect relevant prior candidates, matched case outcomes, and lessons. Choose general mechanisms supported by the evidence; never select using test results.',
 'diagnose':'Read the supplied training evidence and target-visible interactions. Separate task failures from infrastructure faults. Explain the cause in the existing OpenCode instructions/tools/skills and propose a general repair. Do not assume this is a Meta-ARE agent or a GAIA application.',
 'capability-patch':'Implement a reusable OpenCode custom tool or skill when the failure needs a capability. Writable candidate paths are instructions.md, .opencode/tools and .opencode/skills. Preserve the pinned OpenCode API and model. Do not modify benchmark adapters, graders, manifests, provider routing, task deadlines, or hidden tests.',
 'steering-patch':'Improve reusable behavioral instructions or skills for the observed failure. Change instructions.md or .opencode/skills only. Avoid benchmark IDs, reference answers, task-specific constants, and hacks tied to a training instance.',
 'patch-verification':'Inspect the actual diff and changed paths. Check syntax and loading/invocation of changed OpenCode tools or skills. Verify instructions remain nonempty. Return the actual changed_paths in the required JSON schema; do not report an unexecuted check as passed.',
}


class NativeOpenCodePromptPack(MetaAREPromptPack):
    def __init__(self, *, adaptive_reflection_memory=False, optimizer_model='Kimi', **kwargs):
        super().__init__(**kwargs)
        self.adaptive_memory = AdaptiveReflectionMemory(self.store) if adaptive_reflection_memory else None
        self.optimizer_model = optimizer_model

    def session(self,kind,context):
        original=super().session(kind,context)
        system='''You optimize the COMPLETE target OpenCode 1.19.5 source and harness. The target model is frozen Qwen and the optimizer is OPTIMIZER_MODEL. This is an OpenCode experiment, not a Meta-ARE agent implementation.
Follow the supplied training evidence and development-gated search protocol. Preserve candidate identity, model and evaluator boundaries. Mutable candidate components are the entire opencode-src/ tree, instructions.md, opencode.json and .opencode/. You may change the core agent loop, context handling, prompts, tool execution and all target implementation modules. Capability and steering phases describe intent; both permit source changes. Benchmark tasks, evaluators, model IDs, credentials, gateway routing, accounting, external quotas, official deadlines and trusted outer cleanup are immutable and outside this workspace.
Build dependencies are pinned in this isolated container. To check actual compilation, run python3 /autoharness-build-check.py /workspace/opencode-src. This compiles your candidate using fixed dependencies and the official headless Linux build. Do not run package installs, fetch model metadata, change model/provider, or access Docker/host services. Changes to dependency declarations must remain resolvable against the pinned offline dependencies; new external dependencies require infrastructure work outside this search. Compilation or runtime protocol failure is technical NOSCORE, never task score zero.
Read the delivered evidence and relevant history before changing code. A receipt proves which exact files were delivered, not that a proposed mechanism is true. Distinguish observations, hypotheses and demonstrated causes. Check the current parent ID and accepted/rejected gate in session_context/history; rejected patches do not become later parent code. Different outcomes from identical bytes are variability, not proof of a source effect. Do not claim local checks or root causes without actual evidence.
Read .autosaddler/target_runtime_defaults.json. H0 preserves the effective original Go context defaults (compaction enabled, input32000, saved complete tool outputs). You can replace ALL those target defaults: create .opencode/runtime-policy.json with exactly {"schema_version":"opencode-full-source-runtime-policy/v1","use_baseline_context_policy":false}, then control context in your native config, source and candidate plugins. This is a mutable target policy switch, not a restriction on core-source changes. The external accounting/model/quota/verifier boundaries remain fixed.
Read the local session context and the relevant skills before acting. Use workspace-scoped paths and container-local /tmp for scratch; do not glob the container root or request host files. Tool excerpts point to complete output files that may be read in bounded ranges.
Produce the required structured object in .autosaddler/session_output.json. Validate it against .autosaddler/session_output_schema.json before ending. Retain the schema_version exactly as supplied; its historical name does not change the target harness.
All changed_paths entries must be exact workspace-relative POSIX paths. Use "instructions.md" and ".opencode/tools/example.ts", never "/workspace/instructions.md", "./instructions.md", Windows paths or parent traversal. Include every actually changed candidate file, including deletions; omit unchanged files. Do not rewrite a file merely to make its path appear in the list.
An objectively rejected patch is allowed. Never fabricate verification, training outcomes, changed files, or missing evidence.'''
        system=system.replace('OPTIMIZER_MODEL',self.optimizer_model)
        tasks={
            'evolve':'Read the context and history. Select allowed parent candidates and component sources using accumulated development evidence and lessons. Write the required evolution JSON; do not modify candidate files.',
            'diagnose_patch':'Analyze the current training failure evidence and existing OpenCode harness. Use history-analysis, diagnose, the selected patch-phase skill and patch-verification. Apply and locally verify a general candidate change, then write the required diagnosis JSON with the actual changed paths.',
            'reflect':'Compare the supplied matched before/after training outcomes and development gate. Record evidence-grounded reusable lessons with valid case IDs in the required reflection JSON. Do not change candidate files.',
        }
        skills={name:f'---\nname: {name}\ndescription: "OpenCode optimization: {name}"\n---\n\n# {name}\n\n{GUIDES[name]}\n' for name in original.skills}
        files=dict(original.workspace_files)
        schema=to_json_value(original.output_schema)
        task_prompt=tasks[kind]
        if kind == 'diagnose_patch':
            schema['properties']['changed_paths']['items'] = {
                'type': 'string',
                'pattern': r'^(?!.*(?:^|/)\.\.?(?:/|$))(?:instructions\.md|opencode\.json|(?:\.opencode|opencode-src)/[^\\:]+)$'
            }
            task_prompt += ('\nBefore ending, report exact workspace-relative paths in changed_paths. '
                            'The transport checks this against actual before/after candidate bytes within the same session; '
                            'the unchanged core independently checks the complete workspace delta afterward.')
        if kind == 'reflect':
            train_ids=context.get('train_case_ids')
            if not isinstance(train_ids,(list,tuple)) or not train_ids or any(not isinstance(x,str) or not x for x in train_ids):
                raise ValueError('Reflection requires explicit current train_case_ids')
            if len(set(train_ids)) != len(train_ids):
                raise ValueError('Reflection train_case_ids must be unique')
            allowed=list(train_ids)
            schema['properties']['lessons']['items']['properties']['evidence_case_ids']['items']={'type':'string','enum':allowed}
            task_prompt += ('\nEvery evidence_case_ids entry must come from this current training batch only: '
                            +json.dumps(allowed)+'. History may inform your reasoning, but IDs from earlier batches, '
                            'development or test are not valid evidence_case_ids for this reflection. '
                            'Use only supplied matched evidence; do not fabricate or relabel evidence to satisfy the schema.')
            files['.autosaddler/reflection_evidence_scope.json']=json.dumps({'train_case_ids':allowed},sort_keys=True)
        if self.adaptive_memory is not None:
            memory = self.adaptive_memory
            stages = ('diagnose_patch', 'patch') if kind == 'diagnose_patch' else (kind,)
            for stage in stages:
                retrieval = memory.retrieve(stage, context)
                files[f'.autosaddler/adaptive_memory/{stage}.json'] = json.dumps(retrieval,sort_keys=True)
                files.update(memory.selected_details(retrieval))
            system += ('\nH2 uses an experiment journal. Read the stage-specific files in '
                       '.autosaddler/adaptive_memory/ and inspect their source_operation_id events or full '
                       '.autosaddler/history/ entries before relying on a compressed claim. '
                       'A prior accepted patch is not proof of its stated mechanism. Never use test data.')
            if kind == 'evolve':
                task_prompt += ('\nUse retrieved candidate history, durable fixes and regressions to choose parents. '
                                'Explain which prior observation supports the choice.')
            elif kind == 'diagnose_patch':
                task_prompt += ('\nFirst diagnose the failing trajectory using retrieved similar failures and rejected hypotheses. '
                                'Then consider a source-code repair in the active OpenCode execution path: tool implementation, '
                                'schema, routing, result/error handling, call state or reliability. '
                                'Change a tool only when the observed failure and causal path support it. '
                                'Verify the actual diff and compile; do not make a cosmetic tool edit.')
            else:
                task_prompt += ('\nCompare exact before/after task behavior with the changed active code path. '
                                'Classify the hypothesis as confirmed, refuted or uncertain. '
                                'A metric change alone is not causal evidence; preserve uncertainty for stochastic outcomes '
                                'or multi-component patches. Give a concrete next step and source references.')
                schema['properties']['causal_assessment'] = {
                    'type':'object', 'additionalProperties':False,
                    'required':['status','behavioral_mechanism','evidence','evidence_kind','evidence_case_ids','source_refs','next_step'],
                    'properties': {
                        'status':{'type':'string','enum':['confirmed','refuted','uncertain']},
                        'behavioral_mechanism':{'type':'string'},
                        'evidence':{'type':'string'},
                        'evidence_kind':{'type':'string','enum':['trace','code_route','tool_behavior','score_only','insufficient']},
                        'evidence_case_ids':{'type':'array','items':{'type':'string','enum':allowed},'uniqueItems':True},
                        'source_refs':{'type':'array','items':{'type':'string'},'uniqueItems':True},
                        'next_step':{'type':'string','minLength':1},
                    },
                    'allOf':[{'if':{'properties':{'status':{'const':'confirmed'}}},
                              'then':{'properties':{'evidence_kind':{'enum':['trace','code_route','tool_behavior']},
                                                    'evidence_case_ids':{'minItems':1},'source_refs':{'minItems':1}}}}],
                }
                schema['required'].append('causal_assessment')
        files['.autosaddler/opencode_prompt_provenance.json']=json.dumps(dict(version=VERSION,kind=kind,
            skill_sha256={name:hashlib.sha256(text.encode()).hexdigest() for name,text in skills.items()},
            output_schema_sha256=hashlib.sha256(json.dumps(schema,sort_keys=True).encode()).hexdigest()),sort_keys=True)
        # Replace old path-limited guides as well as the top-level instruction.
        for name in skills:
            guide = {
                'history-analysis': 'Read current context and history. Distinguish accepted parents from rejected proposals. Use development only for candidate selection; no test access.',
                'diagnose': 'Read complete delivered train evidence and interactions. Locate the general mechanism in the actual target source. Mark uncertain causes as hypotheses.',
                'capability-patch': 'Implement a general capability in any target source module, configuration, tool or skill. Do not encode task IDs, reference answers, or task-specific constants.',
                'steering-patch': 'Improve general target behavior in its source, loop, context policy, instructions or tools. All target source is mutable; immutable evaluation/model/accounting boundaries still apply.',
                'patch-verification': 'Inspect the exact candidate diff and compile source changes using the supplied offline build checker. Report actual relative changed paths and actual checks. Do not claim checks passed if unexecuted.',
            }[name]
            skills[name] = f'---\nname: {name}\ndescription: "Full-source OpenCode optimization: {name}"\n---\n\n# {name}\n\n{guide}\n'
        if self.adaptive_memory is not None and kind == 'diagnose_patch':
            for name in ('diagnose','capability-patch','steering-patch','patch-verification'):
                if name in skills:
                    skills[name] += ('\nH2: inspect the actual active source route for the observed tool failure. '
                                     'Prefer a verified source repair when justified by the trace. '
                                     'Record prior failed attempts and do not edit tools merely to satisfy a preference.\n')
        files['.autosaddler/opencode_prompt_provenance.json'] = json.dumps({
            'version': VERSION, 'kind': kind,
            'skill_sha256': {name: hashlib.sha256(text.encode()).hexdigest() for name, text in skills.items()},
            'output_schema_without_receipt_sha256': hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest(),
        }, sort_keys=True)
        from .target_policy import DESCRIPTION
        files['.autosaddler/target_runtime_defaults.json'] = json.dumps(DESCRIPTION, sort_keys=True)
        delivered = {name: hashlib.sha256(content.encode('utf-8')).hexdigest()
                     for name, content in sorted(files.items())}
        files['.autosaddler/evidence_index.json'] = json.dumps({
            'kind': kind, 'files': delivered, 'evidence_scope': 'train/development history only; no test',
            'interpretation': 'observed variation is not a proven causal source effect'}, sort_keys=True)
        schema['properties']['evidence_receipt'] = {
            'type': 'object', 'properties': {name: {'type': 'string', 'const': digest}
                for name, digest in delivered.items()}, 'required': list(delivered), 'additionalProperties': False}
        schema.setdefault('required', []).append('evidence_receipt')
        task_prompt += '\nRead .autosaddler/evidence_index.json and its current evidence/history. Include its exact files map as evidence_receipt in your structured result.'
        return replace(original,system_context=system,task_prompt=task_prompt,skills=skills,workspace_files=files,output_schema=schema)
