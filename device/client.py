import json
from paramiko import SSHClient
from scp import SCPClient
from chronometre import chronometre
from logger import logger


_config_path = 'configs/ssh_config.json'
with open(_config_path) as f:
    _config = json.load(f)

def ssh_exec_cmd(cmd, max_trial=5):
    """
    Based on short connection.
    """
    for _ in range(max_trial):
        client = None
        try:
            client = _create_ssh_client(**_config)
            stdin, stdout, stderr = client.exec_command(cmd)
            stdin.close()
            result = stdout.read()
            stdout.close()
            err = stderr.read()
            stderr.close()
            return result, err
        except Exception as e:
            logger.error(f'Error occurs when sending cmd: {e}. Retry.')
        finally:
            if client is not None:
                client.close()

def scp_push_file(src_path, dst_path, max_trial=5):
    """
    Based on short connection.
    """
    for _ in range(max_trial):
        ssh, scp = None, None
        try:
            ssh = _create_ssh_client(**_config)
            scp = _create_scp_client(ssh)
            scp.put(src_path, dst_path)
            return True
        except Exception as e:
            logger.error(f'Exception occurs: {e}. Retry.')
        finally:
            if ssh is not None:
                ssh.close()
            if scp is not None:
                scp.close()
    return False

class Client:
    """
    An SSH/SCP client based on long connection.
    """
    def __init__(self, config_path='ssh_config.json'):
        with open(config_path) as f:
            config = json.load(f)
        self.ssh = _create_ssh_client(**config)
        self.scp = _create_scp_client(self.ssh)

    def exec_cmd(self, cmd):
        stdin, stdout, stderr = self.ssh.exec_command(cmd)
        stdin.close()
        result = stdout.read()
        stdout.close()
        err = stderr.read()
        stderr.close()
        return result, err

    def push(self, src_path, dst_path):
        self.scp.put(src_path, dst_path)

def _create_ssh_client(ip, port, username, password, timeout=10):
    client = SSHClient()
    client.load_system_host_keys()
    client.connect(ip, 
                   port=port,
                   username=username,
                   password=password,
                   timeout=timeout)
    return client

def _create_scp_client(ssh_client):
    return SCPClient(ssh_client.get_transport())

