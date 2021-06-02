# Define an inference configuration
from azureml.core.webservice import LocalWebservice
from azureml.core import Environment
from azureml.core.model import InferenceConfig
from azureml.core.model import Model
from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.core.webservice import AciWebservice
import resources.azure_messaging_config as conf

interactive_auth = InteractiveLoginAuthentication(tenant_id=conf.TENANT_ID)
ws = Workspace.from_config(auth=interactive_auth)

env = Environment.from_conda_specification(
        name='pytorch-env',
        file_path='../.azureml/pytorch-env.yml'
        )

model = Model(ws, name='short_text_model') # name is the name of your registered model

service_name = "mynewservice" # deployment service name

inference_config = InferenceConfig(environment=env, entry_script='./echo_score.py')

deployment_config = AciWebservice.deploy_configuration(cpu_cores = 1, memory_gb = 2)

# Deploy your machine learning model
service = Model.deploy(ws, service_name, [model], inference_config, deployment_config, overwrite=True)
service.wait_for_deployment(show_output=True)
print(service.get_logs())
print(service.scoring_uri)