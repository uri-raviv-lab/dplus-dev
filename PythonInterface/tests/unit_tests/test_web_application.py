import hashlib
import json
import tempfile
import uuid
import os
import pytest
import requests

from dplus.CalculationInput import CalculationInput
from dplus.CalculationRunner import WebRunner
from tests.old_stuff.fix_state_files import fix_file
from tests.unit_tests.conftest import test_dir

token="02ab7070a0412fc1a7f7d400931529ad50f3535d"#"fd8d997a007d96a320d556846a48d340315faf2e"
base_url="http://192.168.18.100/"#"http://localhost:8000/"
runner=WebRunner(base_url, token)

def _calc_data():
    state_file=os.path.join(test_dir, "pdbstate.state")
    fixed_state_file = fix_file(state_file)
    calc_data=CalculationInput.load_from_state_file(fixed_state_file)
    return calc_data

@pytest.mark.incremental
class TestWebApp():
    _url = base_url + r'api/v1/'
    _header = {'Authorization': "Token " + str(token)}
    _session = str(uuid.uuid4())
    _session_string = "?ID=" + _session
    files_missing = []

    def test_get_metadata(self):
        metadata = requests.get(url=self._url + 'metadata', headers=self._header)
        if metadata.status_code == 200:
            result = json.loads(metadata.text)
            print("requested job status, results were", result)

    def test_check_files(self):
        calc_data=_calc_data()
        def calc_file_hash(filepath):  # taken from stackexchange 1869885/calculating-sha1-of-a-file
            sha = hashlib.sha1()
            with open(filepath, 'rb') as f:
                while True:
                    block = f.read(2 ** 10)  # Magic number: one-megabyte blocks.
                    if not block: break
                    sha.update(block)
                return sha.hexdigest().upper()

        filedict = {'files': []}
        for filename in calc_data._filenames:
            dict = {}
            dict['filename'] = filename
            dict['size'] = os.stat(filename).st_size
            dict['hash'] = calc_file_hash(filename)
            filedict['files'].append(dict)
        data = json.dumps(filedict)
        response = requests.post(url=self._url + 'files', data=data, headers=self._header)
        statuses = json.loads(response.text)

        for filename in calc_data._filenames:
            if statuses[filename]['status'] == 'MISSING':
                self.files_missing.append((filename, statuses[filename]['id']))


    def test_upload_files(self):
        for filename, id in self.files_missing:
            url = self._url + 'files/' + str(id)
            files = {'file': open(filename, 'rb')}
            response = requests.post(url=url, headers=self._header, files=files)

    def test_start_generate(self):
        calc_data=_calc_data()
        data = json.dumps(calc_data.args['args'])
        test = requests.put(url=self._url + "generate" + self._session_string, data=data, headers=self._header)
        print(test.text)
        assert test.status_code==200

    def test_get_job_status(self):
        finished=False
        while not finished:
            jobstatus = requests.get(url=self._url + 'job' + self._session_string, headers=self._header)
            if jobstatus.status_code == 200:
                result = json.loads(jobstatus.text)
                assert result["error"]["message"]=="OK"
                finished=not result["result"]["isRunning"]
            else:
                raise Exception("Job status code not 200")

    def test_get_generate_results(self):
        response = requests.get(url=self._url + "generate" + self._session_string, headers=self._header)
        result = json.loads(response.text)
        assert len(result["result"]["Graph"])

    def test_get_amp(self):
        model_ptr=3
        destination_folder = tempfile.mkdtemp()
        ptr_string = '%08d.amp' % (int(model_ptr))
        destination_file = os.path.join(destination_folder, ptr_string)
        # code used from: https://stackoverflow.com/questions/13137817/how-to-download-image-using-requests
        response = requests.get(url=self._url + "amplitude/" + str(model_ptr) + self._session_string,
                                headers=self._header, stream=True)
        with open(destination_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
        with open(destination_file, 'rb') as f:
            assert f.read()

    def test_get_pdb(self):
        model_ptr=2
        destination_folder = tempfile.mkdtemp()
        ptr_string = '%08d.pdb' % (int(model_ptr))
        destination_file = os.path.join(destination_folder, ptr_string)
        # code used from: https://stackoverflow.com/a/16696317/5961793
        response = requests.get(url=self._url + "pdb/" + str(model_ptr) + self._session_string, headers=self._header,
                                stream=True)
        with open(destination_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
        with open(destination_file, 'rb') as f:
            assert f.read()


    def test_run_fit(self):
        result=runner.fit(_calc_data())
        assert len(result.graph)