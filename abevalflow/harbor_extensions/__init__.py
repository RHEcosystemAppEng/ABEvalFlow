"""Harbor environment extensions for Agentic Eval Flow OpenShift deployment.

Includes OpenShiftEnvironment emptyDir mounts and AEH task enrichment
(skills + annotations) applied before ``harbor run``.

Runtime contract for classic Harbor A/B and AEH Harbor runs:
- ``harbor==0.20.0`` (PyPI; stock upstream, not skills_eval_corrections)
- ``kubernetes>=32.0.0``
- AEH ``agent_eval.harbor.kubernetes.KubernetesEnvironment`` on PYTHONPATH
- ``abevalflow`` on PYTHONPATH (this package)

Custom OpenShift env is selected only via
``OPENSHIFT_ENVIRONMENT_IMPORT_PATH`` / ``--environment-import-path``.
Prebuilt trial images are passed as task.toml ``docker_image``.
Cluster ask: create/get/delete pods, exec, pull secrets; no harbor-task-scc.
"""

# Shared import path for Harbor --environment-import-path (classic + AEH).
OPENSHIFT_ENVIRONMENT_IMPORT_PATH = "abevalflow.harbor_extensions.openshift_environment:OpenShiftEnvironment"

__all__ = ["OPENSHIFT_ENVIRONMENT_IMPORT_PATH"]
