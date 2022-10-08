import json
import boto3
import os

def dd_handler(event, context):
    runtime = boto3.Session().client('sagemaker-runtime')
    endpoint_name = 'triage-model'
    endpoint_name = os.environ["ENDPOINT_NAME"]
    response = runtime.invoke_endpoint(EndpointName = endpoint_name,# The name of the endpoint we created
                                       ContentType = 'application/json',                 # The data format that is expected
                                       Body = json.dumps(event))
                                       
    return json.loads(response["Body"].read().decode('utf-8'))