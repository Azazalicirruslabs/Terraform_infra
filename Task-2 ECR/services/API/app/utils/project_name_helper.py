import os

from dotenv import load_dotenv

load_dotenv()

S3_BUCKET = os.getenv("S3_BUCKET")


def get_next_project_folder(s3_client, tenant_name: str, username: str, analysis: str) -> str:
    """
    Find the next available project folder under analysis.
    Example path: tenant_name/username/analysis/project_1/
    """
    prefix_base = f"{tenant_name}/{username}/{analysis}/"
    response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix_base, Delimiter="/")

    existing_projects = set()
    if "CommonPrefixes" in response:
        for cp in response["CommonPrefixes"]:
            folder = cp["Prefix"].replace(prefix_base, "").split("/")[0]
            if folder.startswith("project_"):
                try:
                    num = int(folder.replace("project_", ""))
                    existing_projects.add(num)
                except ValueError:
                    continue

    next_num = 1
    while next_num in existing_projects:
        next_num += 1

    return f"project_{next_num}"
