from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.train.hyperdrive import GridParameterSampling
from azureml.train.hyperdrive import HyperDriveConfig
from azureml.core import Dataset
from azureml.widgets import RunDetails
from azureml.train.hyperdrive.parameter_expressions import choice
from azureml.train.hyperdrive import PrimaryMetricGoal
from azureml.core.resource_configuration import ResourceConfiguration
from azureml.train.hyperdrive import MedianStoppingPolicy

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
        script='train.py',
        compute_target='cpu-cluster',
        arguments=[
            '--data_path', dataset.as_named_input('input').as_mount()],
    )
    # set up pytorch environment
    env = Environment.from_conda_specification(
        name='pytorch-env',
        file_path='./.azureml/pytorch-env.yml'
    )
    config.run_config.environment = env

    param_sampling = GridParameterSampling({
        'batch_size': choice(32),
        'learning_rate': choice(0.1),
        'adam_epsilon': choice(1e-8),
        'num_epochs': choice(5)})

    # early_termination_policy = BanditPolicy(slack_factor=0.15, evaluation_interval=1, delay_evaluation=10)
    early_termination_policy = MedianStoppingPolicy(evaluation_interval=1, delay_evaluation=3)

    hyperdrive_run_config = HyperDriveConfig(run_config = config,
                                         hyperparameter_sampling = param_sampling,
                                         policy = early_termination_policy,
                                         primary_metric_name = 'validation_loss',
                                         primary_metric_goal = PrimaryMetricGoal.MINIMIZE,
                                         max_total_runs = 8,
                                         max_concurrent_runs = 4)

    run = experiment.submit(config)
    # run = experiment.submit(hyperdrive_run_config)
    run.wait_for_completion(show_output=True)

    assert(run.get_status() == "Completed")

    # best_run = run.get_best_run_by_primary_metric()
    # best_run_metrics = best_run.get_metrics()
    # print(best_run)
    # print(best_run_metrics)

    # print('Best Run is:\n  Validation accuracy: {0:.5f} \n  Learning rate: {1:.5f} \n  Momentum: {2:.5f}'.format(
    #     best_run_metrics['best_f1'][-1],
    #     best_run_metrics['lr'],
    #     best_run_metrics['momentum'])
    #  )

    # Model generated.. Registering it next..

    out_dir = './outputs'

    model = run.register_model(model_name='short_text_model',
                               tags={'area': 'qna'},
                               model_path=out_dir + '/saved_weights.pt',
                               resource_configuration=ResourceConfiguration(cpu=1, memory_in_gb=2))

    print("Model '{}' version {} registered ".format(model.name,model.version))

    model.download("./outputs", exist_ok=True)
