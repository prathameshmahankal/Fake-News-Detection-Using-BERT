from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.core import Dataset
from azureml.widgets import RunDetails

if __name__ == "__main__":

    """
    This script sets up the workspace and the experiment 
    and submits the experiment once it is ready
    """

    # ws = Workspace.from_config()
    interactive_auth = InteractiveLoginAuthentication(tenant_id="72f988bf-86f1-41af-91ab-2d7cd011db47")
    ws = Workspace.from_config(auth=interactive_auth)

    datastore = ws.get_default_datastore()
    dataset = Dataset.File.from_files(path=(datastore, 'datasets/cifar10'))

    experiment = Experiment(workspace=ws, name='day1-experiment-data')

    # A ScriptRunConfig packages together the configuration information needed 
    # to submit a run in Azure ML, including the script, compute target, 
    # environment, and any distributed job-specific configs.
    
    config = ScriptRunConfig(
        source_directory='./src',
        script='trainShort.py',
        compute_target='cpu-cluster',
        arguments=[
            '--data_path', dataset.as_named_input('input').as_mount(),
            '--learning_rate', 0.003],
    )
    # set up pytorch environment
    env = Environment.from_conda_specification(
        name='pytorch-env',
        file_path='./.azureml/pytorch-env.yml'
    )
    config.run_config.environment = env

    run = experiment.submit(config)
    # aml_url = run.get_portal_url()
    print("Submitted to compute cluster. Click link below")
