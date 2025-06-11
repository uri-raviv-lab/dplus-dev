using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

/// <summary>
/// @file CSharpManagedBackendCall.cs
/// @brief Defines a data structure for representing backend function calls and their results in the PythonBackend system.
///
/// The CSharpManagedBackendCall class is responsible for:
///  - Encapsulating all information related to a backend function call, including the function name, arguments, client identifier, and options.
///  - Storing the result, error code, and error message associated with the backend call.
///  - Providing property accessors for all fields to facilitate serialization, deserialization, and inter-process communication.
///
/// Key Concepts:
///  - Backend Call Representation: Serves as a container for all data needed to describe and track a backend function invocation.
///  - Error and Result Handling: Stores both the result of the call and any error information for robust communication between components.
///  - Interoperability: Designed for use in managed (.NET) environments, often in scenarios involving Python or other external clients.
///
/// Fields:
///  - string CallString: The full string representation of the backend call.
///  - string FuncName: The name of the function to be called.
///  - string Args: The arguments to the function, typically serialized.
///  - string ClientId: An identifier for the client making the call.
///  - int ErrorCode: Numeric error code resulting from the call.
///  - string ErrorMessage: Error message, if any, resulting from the call.
///  - string Result: The result of the backend call, typically serialized.
///  - string Options: Additional options or metadata for the call.
///
/// Main Methods:
///  - CSharpManagedBackendCall(): Initializes all fields to default values.
///  - Property accessors for all fields (getters and setters).
///
/// Error Handling:
///  - ErrorCode and ErrorMessage fields are used to communicate failure states and diagnostic information.
///
/// Dependencies:
///  - System namespace for basic .NET types.
///  - No external dependencies beyond the .NET Framework.
///
/// See also:
///  - PythonBackendCaller.cs for usage and integration with backend call dispatch.
/// </summary>

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
