from pathlib import Path

import boto3
import rasterio as rio
from cognito_client import CognitoClient
from rasterio.session import AWSSession

client = CognitoClient(
    identity_pool_id="us-west-2:XXX",
    user_pool_id="us-west-2_XXX",
    client_id="XXX",
)
_ = client.login()

# Fetch AWS Credentials
creds = client.get_aws_credentials()

aws_access_key_id = creds["AccessKeyId"]
aws_secret_access_key = creds["SecretKey"]
aws_session_token = creds["SessionToken"]

session = boto3.Session(
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    aws_session_token=aws_session_token,
)

if __name__ == "__main__":
    rio_env = rio.Env(
        AWSSession(session),
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        GDAL_HTTP_COOKIEFILE=str(Path("~/cookies.txt").expanduser()),
        GDAL_HTTP_COOKIEJAR=str(Path("~/cookies.txt").expanduser()),
    )
    rio_env.__enter__()
