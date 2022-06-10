import subprocess
import json

def dataverse_upload(
        file_path,
        dataverse_url,
        dataverse_token,
        dataverse_dir,
        dataverse_id,
        log,
        retry=10):
    '''
    Upload a file to dataverse
    '''
    cmd = [
        "curl",
        "-X", "POST",
        "-H", f"X-Dataverse-key:{dataverse_token}",
        "-F", f"file=@{file_path}",
        "-F", 'jsonData='+json.dumps({
            "description":"",
            "directoryLabel":f"{dataverse_dir}/",
            "categories":["Data"],
            "restrict": "false"
        }),
        f"{dataverse_url}/api/datasets/:persistentId/add?persistentId={dataverse_id}",
        "--retry", str(retry)]
    log.info(cmd)
    subprocess.Popen(cmd)