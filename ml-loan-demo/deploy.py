from mlflow.deployments import get_deploy_client

RUN_ID = "a1876b7c97c94604a58d7bb4686974f4"
model_uri = f"runs:/{RUN_ID}/model"

EXECUTION_ROLE_ARN = "arn:aws:iam::750952118292:role/SagemakeDemo"

client = get_deploy_client("sagemaker")
client.create_deployment(
    name= "loan-regression-app",
    model_uri= model_uri,
    config={
        "region_name":   "eu-west-2",  
        "instance_type": "ml.m5.large",
        "instance_count":"1",
         "execution_role_arn" : EXECUTION_ROLE_ARN
    }
)

print("🚀 Deployed to SageMaker endpoint: loan-regression-app")
