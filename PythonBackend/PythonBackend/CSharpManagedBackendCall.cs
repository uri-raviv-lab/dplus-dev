using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace PythonBackend
{
    public class CSharpManagedBackendCall
    {
        string _callString;
        string _funcName;
        string _args;
        string _clientId;
        int _errorCode;
        string _errorMessage;
        string _result;
        string _options;
        public CSharpManagedBackendCall()
        {
            _callString = "";
            _funcName = "";
            _args = "";
            _clientId = "";
            _errorCode = 0;
            _errorMessage = "";
            _result = "";
            _options = "";
        }
        public string CallString { get => _callString; set => _callString = value; }
        public string FuncName { get => _funcName; set => _funcName = value; }
        public string Args { get => _args; set => _args = value; }
        public string ClientId { get => _clientId; set => _clientId = value; }
        public int ErrorCode { get => _errorCode; set => _errorCode = value; }
        public string ErrorMessage { get => _errorMessage; set => _errorMessage = value; }
        public string Result { get => _result; set => _result = value; }
        public string Options { get => _options; set => _options = value; }
    }



}
