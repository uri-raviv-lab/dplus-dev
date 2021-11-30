#include "ManagedHTTPCallerForm.h"
#include "../../Frontend/Frontend/BackendCalls.h"
#include <msclr\marshal_cppstd.h>
#include <string>

namespace DPlus{


	public ref class WebCallException : Exception
	{
	public:
		WebCallException()
		{
			_errorCode = 0;
		}

		WebCallException(String^ message): Exception(message)
		{
		}
		WebCallException(ErrorCode err, String^  error_message) :Exception(error_message)
		{
			_errorCode = err;
		}

		WebCallException(String^ message, Exception^ inner): Exception(message, inner)
		{
			_errorCode = 0;
		}
		int GetErrorCode()
		{
			return _errorCode;
		}
	private:
		int _errorCode;
	};

	void ManagedHTTPCallerForm::PerformCall(BackendCall &_call, bool runInBackground)
	{

		if (!runInBackground) {
			//populate call and postData
			class_call = &_call; //make call available to the class, so dialogue launch has access to it
			this->ShowDialog();
		}

		else
		{
			BackendCall * single_call = &_call;
			executeHttp(single_call);
		}
	}

	System::Void ManagedHTTPCallerForm::On_Shown(System::Object^  sender, System::EventArgs^  e)
	{
		BGWCall();
	}


	void ManagedHTTPCallerForm::OnDoWork(System::Object ^sender, System::ComponentModel::DoWorkEventArgs ^e)
	{
		BackendCall * single_call = class_call;
		executeHttp(single_call);
	}

	void ManagedHTTPCallerForm::BGWCall()
	{
		//setup BackgroundWorker
		BackgroundWorker ^ bgw = gcnew BackgroundWorker();
		bgw->DoWork += gcnew System::ComponentModel::DoWorkEventHandler(this, &DPlus::ManagedHTTPCallerForm::OnDoWork);
		bgw->RunWorkerCompleted += gcnew System::ComponentModel::RunWorkerCompletedEventHandler(this, &DPlus::ManagedHTTPCallerForm::OnRunWorkerCompleted);
		bgw->RunWorkerAsync();

	}


	String ^ ManagedHTTPCallerForm::httpCall(String ^ _URLnoID, String^ postData, String ^ method)
	{
		try
		{
			// Open a connection
			String ^ _URL = _URLnoID + "?ID=" + sessionID;
			HttpWebRequest ^_HttpWebRequest = safe_cast<HttpWebRequest^>(HttpWebRequest::Create(_URL));
			_HttpWebRequest->Headers->Add("Authorization", "Token " + token);
			_HttpWebRequest->Method = method;
			_HttpWebRequest->KeepAlive = false;




			if ((String::Compare(method, "POST") == 0 || (String::Compare(method, "PUT") == 0)))
			{
				//for post and put
				UTF8Encoding ^ encoding = gcnew UTF8Encoding;
				array<Byte>^ byte1 = encoding->GetBytes(postData);

				_HttpWebRequest->AllowWriteStreamBuffering = true;

				_HttpWebRequest->ContentType = "application/json";
				_HttpWebRequest->ContentLength = byte1->Length;

				Stream^ newStream = _HttpWebRequest->GetRequestStream();
				newStream->Write(byte1, 0, byte1->Length);
				newStream->Close();
			}

			// Request response:
			WebResponse ^_WebResponse = _HttpWebRequest->GetResponse();
			// Open data stream:
			Stream ^_WebStream = _WebResponse->GetResponseStream();
			StreamReader ^ readStream = gcnew StreamReader(_WebStream);
			String ^managedResponse = readStream->ReadToEnd();

			// Cleanup

			_WebResponse->Close();

			return managedResponse;
		}

		catch (WebException ^ we)
		{
			String ^managedResponse;
			
			try
				{
					managedResponse = StreamReader(we->Response->GetResponseStream()).ReadToEnd();
				}
			
			catch (Exception ^)
				{
					throw(gcnew WebCallException(ERROR_NO_RESPONSE, "")); //no response from server-- see error codes in common.h
				}

			
			//we have successfully received response from server
			
			//check for error in json:
			try
			{
				JObject ^ j_item = JObject::Parse(managedResponse);

				JToken ^ exist_error;
				exist_error = j_item->TryGetValue("error", exist_error);
				if (exist_error)
				{
					return managedResponse; //D+ knows how to handle errors generated in json, just send it back.
				}

				else
				{
					JToken ^ exist_detail = j_item->TryGetValue("detail", exist_detail);
					if (exist_detail)
					{
						JToken ^ _status = j_item["detail"];
						String ^ detail = _status->ToString();
						if (detail == "Invalid token.")
						{
							tokenfailed = true;
							throw(gcnew WebCallException(ERROR_BAD_TOKEN, "")); //"Invalid activation code"
						}
						throw(gcnew WebCallException(ERROR_BAD_TOKEN, detail));
					}
					throw(gcnew WebCallException(ERROR_INTERNAL_SERVER, "")); //internal server error
				}
			}
			catch (WebCallException ^ e)
			{
				throw e;
			}
			catch (Exception ^ e)
			{
				throw gcnew Exception("Server not responding");
			}			
		}
		catch (Exception ^ e)
		{
			throw(e);
		}
	}

	void ManagedHTTPCallerForm::DownloadFile(String^ _URLnoID, BackendCall *call)
	{
		String ^ url = _URLnoID + "?ID=" + sessionID;
		GetFileCall* pF = dynamic_cast<GetFileCall *> (call);
		String ^ filepath = gcnew String(pF->GetFilename().c_str());
		WebClient ^client = gcnew WebClient();
		client->Headers->Add("Authorization", "Token " + token);
		client->DownloadFile(url, filepath);
	}
	enum string_code {
		metadata,
		start_generate,
		get_generate,
		start_fit,
		get_fit,
		job,
		stop,
		pdb,
		amplitude
	};
	string_code hashit(String ^ inString) {
		if (inString == "GetAllModelMetadata") return metadata;
		if (inString == "StartGenerate") return start_generate;
		if (inString == "GetGenerateResults") return get_generate;
		if (inString == "StartFit") return start_fit;
		if (inString == "GetFitResults") return get_fit;
		if (inString == "GetJobStatus") return job;
		if (inString == "GetPDB") return pdb;
		if (inString == "GetAmplitude") return amplitude;
		if (inString == "Stop") return stop;
	}
	void ManagedHTTPCallerForm::executeHttp(BackendCall * call)
	{
		try
		{

			std::string fcn = call->GetFuncName();
			String ^ funcName = gcnew String(fcn.c_str());
			std::string st = call->GetArgs();
			String ^ postData = gcnew String(st.c_str());



			String ^ managedResponse;

			syncFilesWithServer(call);

			String ^ url = base_url;

			switch (hashit(funcName))
			{
			case metadata:
				try
				{
					url += "metadata";
					managedResponse = httpCall(url, postData, "GET");
					System::IO::File::WriteAllText("metadata", managedResponse);
				}
				catch (Exception ^ e) //any exception whatsoever
				{
					if (System::IO::File::Exists("metadata"))
						managedResponse = System::IO::File::ReadAllText("metadata");
					else
					{
						Exception ^ e =gcnew Exception("Could not load model information from server or local storage. Please check your connection settings.");
						throw e;
					}

				}
				break;
			case start_generate:
				url += "generate";
				managedResponse = httpCall(url, postData, "PUT");
				break;
			case get_generate:
				url += "generate";
				managedResponse = httpCall(url, postData, "GET");
				break;
			case start_fit:
				url += "fit";
				managedResponse = httpCall(url, postData, "PUT");
				break;
			case get_fit:
				url += "fit";
				managedResponse = httpCall(url, postData, "GET");
				break;
			case job:
				url += "job";
				managedResponse = httpCall(url, postData, "GET");
				break;
			case stop:
				url += "job";
				managedResponse = httpCall(url, postData, "DELETE");
				break;
			case pdb:
				url += "pdb/";
				url = buildModelPtrUrl(postData, url);
				DownloadFile(url, call);  // Already saves the file
				return;
			case amplitude:
				url += "amplitude/";
				url = buildModelPtrUrl(postData, url);
				DownloadFile(url, call);  // Already saves the file
				return;
			};



			// The JSON returned from the server contains Infinity for infinite doubles. Rapidjson does not like this, so we need to
			// replace it with a value rapidjson does like, namely: 1.797693134862316e308
			String ^rapidJsonResponse = managedResponse->Replace("Infinity", "1.797693134862316e308");
			String ^ pyrapidJsonResponse = rapidJsonResponse->Replace("\"inf\"", "1.797693134862316e308");
			pyrapidJsonResponse = pyrapidJsonResponse->Replace("\"-inf\"", "-1.797693134862316e308");


			//handle the results
			msclr::interop::marshal_context context;
			std::string standardResponse = context.marshal_as<std::string>(pyrapidJsonResponse);
			call->ParseResults(standardResponse);
		}

		catch (WebCallException ^ e)
		{
			String ^ codestr = e->Message;
			int code = e->GetErrorCode();
			String ^ response = "{\"error\": {\"code\": "+ code + ", \"message\": \""+ codestr +"\"}}";
			msclr::interop::marshal_context context;
			std::string standardResponse = context.marshal_as<std::string>(response);
			call->ParseResults(standardResponse);
		}

		catch (Exception ^ e)
		{
			throw(e);
		}
	}

	String ^ ManagedHTTPCallerForm::buildModelPtrUrl(String ^ postData, String ^ url)
	{
		JObject ^ j_item = JObject::Parse(postData);
		String ^ model_ptr = j_item["model"]->ToString();
		return url + model_ptr;
	}



	void ManagedHTTPCallerForm::syncFilesWithServer(BackendCall * call)
	{
		if (!checkFiles(call))
			uploadFiles();
	}

	bool ManagedHTTPCallerForm::checkFiles(BackendCall *call)
	{
		try
		{
			bool allOK = true;

			FileContainingCall* pF = dynamic_cast<FileContainingCall *> (call);
			if (!pF)
				return true;

			std::vector<std::wstring> files = pF->GetFilenames();
			if (files.empty())
				return true;

			String ^ _URL = base_url + "files";
			String ^ managedResponse;

			JArray ^ jarray = gcnew JArray();

			for (int i = 0; i < files.size(); i++)

			{
				String ^ sourceFileName = gcnew String(files[i].c_str());

				JObject ^ f = gcnew JObject();


				f["filename"] = gcnew JValue(sourceFileName);

				FileInfo ^ fi = gcnew FileInfo(sourceFileName);
				f["size"] = gcnew JValue(fi->Length);


				SHA1Managed ^ shaForStream = gcnew SHA1Managed();
				CryptoStream ^ sourceStream = gcnew CryptoStream(gcnew FileStream(sourceFileName, FileMode::Open, FileAccess::Read), shaForStream, CryptoStreamMode::Read);
				array<Byte> ^ shaHash;

				while (sourceStream->ReadByte() != -1);
				{
					shaHash = shaForStream->Hash;
				}

				sourceStream->Close();

				String ^ hex = BitConverter::ToString(shaHash)->Replace("-", "");

				f["hash"] = gcnew JValue(hex);

				jarray->Add(f);
			}

			JObject ^ o = gcnew JObject();
			o["files"] = jarray;
			String ^ postData = o->ToString();

			managedResponse = httpCall(_URL, postData, "POST");


			JObject ^ items = JObject::Parse(managedResponse);
			JObject ^ _item;
			JToken ^ _id;
			JToken ^ _status;
			for (int i = 0; i < files.size(); i++)
			{
				String ^ sourceFileName = gcnew String(files[i].c_str());
				_item = (JObject^)items[sourceFileName];
				_status = _item["status"];
				_id = _item["id"];
				String ^ status = _status->ToString();
				String ^ id = _id->ToString();
				if (status != "OK")
				{
					allOK = false;
					missingfiles[sourceFileName] =  id;
				}
			}


			return allOK;
		}

		catch (Exception ^ e)
		{
			throw(e);
		}

	}

	bool ManagedHTTPCallerForm::uploadFiles()
	{

		try{
			String ^ _URL = base_url + "files/";
			String ^ URLwithID;
			WebClient ^ wb = gcnew(WebClient);
			String ^ value = "Token " + token;
			wb->Headers->Add("Authorization", value);



			for each(KeyValuePair<String ^, String^> kvp in missingfiles)
			{
				String^ id = kvp.Value;
				String^ file = kvp.Key;
				URLwithID = _URL + id;
				wb->UploadFile(URLwithID, file);
			}

			return true;
		}

		finally
		{
			missingfiles.Clear();
		}

	}

	void ManagedHTTPCallerForm::handleErrors(String ^ e)
	{
		//this->Hide();
		my_setWindowDisplay(true, "error");
		this->errorLabel->Text = e;
		this->BringToFront();
	}

	void ManagedHTTPCallerForm::OnRunWorkerCompleted(System::Object ^sender, System::ComponentModel::RunWorkerCompletedEventArgs ^e)
	{
		//4 types of errors:
		//connection errors
		//http errors
		//json errors
		//error in json --this can be passed on, they can handle it

		if (e->Error)
		{
			_ForcedClose = true;
			if (e->Error->GetType() == SEHException::typeid)
				handleErrors("Invalid response from server");
			else
				handleErrors(e->Error->Message);
		}

		else if (!(this->InvokeRequired)) {
			//this->Hide();
			_ForcedClose = false;
			this->Close();
		}
	}

	System::Void ManagedHTTPCallerForm::cancelButton_Click(System::Object^  sender, System::EventArgs^  e)
	{
			cancelButtonClicked(this, e);	
			this->Close();
	}
	System::Void ManagedHTTPCallerForm::restartButton_Click(System::Object^  sender, System::EventArgs^  e)
	{	
		_ForcedClose = false;	
		restartButtonClicked(this, e);
	}
	System::Void ManagedHTTPCallerForm::ManagedHTTPCallerForm_FormClosed(System::Object^  sender, System::Windows::Forms::FormClosedEventArgs^  e)
	{
		if (_ForcedClose)
		if (e->CloseReason == CloseReason::UserClosing)
			cancelButtonClicked(this, e);
	}
	System::Void ManagedHTTPCallerForm::serverLabel_LinkClicked(System::Object^  sender, System::Windows::Forms::LinkLabelLinkClickedEventArgs^  e) 
	{
			serverLabelClicked(this, e);
	}


	System::Void ManagedHTTPCallerForm::retryButton_Click(System::Object^  sender, System::EventArgs^  e)
	{
		my_setWindowDisplay(false, "Contacting Server");
		BGWCall();
	}

	void ManagedHTTPCallerForm::my_setWindowDisplay(bool error, String ^ job)
	{

		if (error)
		{
			this->Text = "ERROR";
			this->errorLabel->Visible = true;
			this->progressBar1->Visible = false;
			this->currentlydoing->Visible = false;
			this->retryButton->Visible = true;
			this->cancelButton->Visible = true;
			this->restartButton->Visible = true;
			this->serverLabel->Visible = true;
			this->serverLabel->Enabled = true;
		}

		else
		{
			this->Text = "DPlus";
			this->errorLabel->Visible = false;
			this->progressBar1->Visible = true;
			this->currentlydoing->Visible = true;
			this->currentlydoing->Text = job;
			this->retryButton->Visible = false;
			this->cancelButton->Visible = false;
			this->restartButton->Visible = false;
			this->errorLabel->Text = "";
			this->serverLabel->Visible = false;
			this->serverLabel->Enabled = false;
		}

	}

}