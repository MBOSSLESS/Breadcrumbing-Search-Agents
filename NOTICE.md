# Attribution

This project is licensed under [Apache-2.0](LICENSE).

The Research Agent and Search/Visit tools derive from Alibaba-NLP's
[Tongyi DeepResearch](https://github.com/Alibaba-NLP/DeepResearch).
Its license is retained in
[licenses/DeepResearch-APACHE-2.0.txt](licenses/DeepResearch-APACHE-2.0.txt).
Those modules have been modified to support the experiments and interfaces in
this repository. The source headers identify the derived files.

The comparison methods are Search/Visit adaptations of AgentDojo, GEO,
SafeSearch, AgentLAB Objective Drift and Intent Hijacking, AiTM, MAST, and
Evo-Attacker. See [baseline implementations](docs/baselines.md) and the paper
for the adaptation scope.

Intent Hijacking includes prompt text from
[AgentLAB](https://github.com/TanqiuJiang/AgentLAB), licensed under MIT.
The original notice is retained in
[licenses/AgentLAB-MIT.txt](licenses/AgentLAB-MIT.txt).

SafeSearch task contents are obtained separately from their upstream dataset.
The published split indices identify the tasks used in the experiments;
this repository does not redistribute the task records or experiment traces.
