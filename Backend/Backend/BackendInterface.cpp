#include "../../BackendCommunication/LocalCommunication/LocalComm.h"
#include "BackendInterface.h"

// Can be whatever you like (std::vectors and such) as long as the BackendComm class
// is portable.

#include "DllBasedBackendWrapper.h"
#include "LocalBackend.h"

/**
 * @file BackendInterface.cpp
 * @brief Provides factory functions for creating backend communication and backend interface objects,
 *        abstracting the underlying implementation (DLL-based or local) for the backend system.
 *
 * The backend interface module is responsible for:
 *  - Defining portable entry points for backend communication and backend object creation.
 *  - Abstracting the instantiation of backend communication (BackendComm) and backend interface (LocalBackend)
 *    to allow for flexible backend implementations (e.g., DLL-based, local, or remote).
 *  - Integrating with the rest of the backend system via well-defined interfaces.
 *
 * Key Concepts:
 *  - BackendComm: Abstract interface for backend communication, allowing for different implementations
 *    (e.g., DLL-based, local, or remote communication).
 *  - LocalBackend: Concrete backend implementation for local (in-process) backend operations.
 *  - DllBasedBackendWrapper: Implementation of BackendComm that communicates with a backend via a DLL interface.
 *
 * Fields:
 *  - BackendComm (class, from BackendInterface.h):
 *      Abstract base class for backend communication. Defines the interface for sending/receiving commands.
 *  - LocalBackend (class, from LocalBackend.h):
 *      Concrete class implementing backend operations locally (in-process).
 *  - DllBasedBackendWrapper (class, from DllBasedBackendWrapper.h):
 *      Concrete implementation of BackendComm that wraps a DLL-based backend.
 *
 * Main Methods:
 *  - BackendComm* CreateBackendComm():
 *      Factory function that creates and returns a new DllBasedBackendWrapper instance as a BackendComm pointer.
 *  - LocalBackend* CreateLocalBackend():
 *      Factory function that creates and returns a new LocalBackend instance.
 *
 * Threading and Safety:
 *  - No internal state is maintained in this file; thread safety is determined by the underlying backend implementations.
 *
 * Error Handling:
 *  - Factory functions return newly allocated objects or nullptr on failure (if allocation fails).
 *
 * Dependencies:
 *  - DllBasedBackendWrapper.h for DLL-based backend communication.
 *  - LocalBackend.h for local backend implementation.
 *  - BackendInterface.h for interface definitions.
 *  - LocalComm.h for local communication primitives.
 *
 * See BackendInterface.h for interface and class declarations.
 */

BackendComm *CreateBackendComm() {
	return new DllBasedBackendWrapper();
}

LocalBackend *CreateLocalBackend() {
	return new LocalBackend();
}
