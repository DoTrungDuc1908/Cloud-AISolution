from azure.ai.ml import MLClient
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment, Model, Environment, CodeConfiguration, OnlineRequestSettings
from azure.identity import DefaultAzureCredential
import os

# --- ĐIỀN THÔNG TIN CỦA BẠN ---
subscription_id = "2fa78aa8-bc5a-4330-ae04-e3f9fcd98b4c"
resource_group = "verba-sentiment-716e9472-rg"
workspace_name = "ml-model-deploy-us"

# 1. Kết nối
credential = DefaultAzureCredential()
ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)

# 2. Tạo Endpoint (Cái vỏ bên ngoài)
endpoint_name = "wavlm-arousal-endpoint" 
try:
    ml_client.online_endpoints.begin_create_or_update(
        ManagedOnlineEndpoint(name=endpoint_name, auth_mode="key")
    ).result()
    print(f"Endpoint {endpoint_name} đã tạo xong.")
except Exception as e:
    print(f"Endpoint có thể đã tồn tại: {e}")

# 3. Tạo Deployment (Ruột bên trong chứa Model)
print("Đang deploy model... (Mất khoảng 10-15 phút)")
deployment = ManagedOnlineDeployment(
    name="green",
    endpoint_name=endpoint_name,
    model=Model(
        name="wavlm-arousal-model",
        path=".", # Tạm thời để path là thư mục hiện tại, nó sẽ tự kéo model từ HF về trong score.py
        type="custom_model"
    ),
    environment=Environment(
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
        conda_file="conda.yaml"
    ),
    code_configuration=CodeConfiguration(
        code=".",
        scoring_script="score.py"
    ),
    instance_type="Standard_E2s_v3", # CPU là đủ, nếu muốn nhanh hơn dùng GPU (Standard_NC6s_v3)
    instance_count=1,

    request_settings=OnlineRequestSettings(
        request_timeout_ms=180000  # Tăng lên 180 giây (3 phút)
    )
)

ml_client.online_deployments.begin_create_or_update(deployment).result()

# 4. Chuyển traffic
endpoint = ml_client.online_endpoints.get(endpoint_name)
endpoint.traffic = {"green": 100, "blue": 0}
ml_client.online_endpoints.begin_create_or_update(endpoint).result()

print("------------------------------------------------")
print(f"DEPLOY THÀNH CÔNG!")
print(f"Scoring URI: {endpoint.scoring_uri}")
print(f"Key (dùng để gọi API): {ml_client.online_endpoints.get_keys(endpoint_name).primary_key}")