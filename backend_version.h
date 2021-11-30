//
// This files contains the version of the backend.
//
// It will be updated manually and replaces the old gitrev.h.
//

#ifndef BACKEND_VERSION_H
#define BACKEND_VERSION_H

// The version is used in RC files for resources, so it must be broken into elements
#define BACKEND_VERSION_MAJOR 4
#define BACKEND_VERSION_MINOR 5
#define BACKEND_VERSION_REVISION 0
// Don't use this field for creating different installer versions as
// WiX (and MSI) MSI specifically ignore this field when comparing versions.
// This leads to many versions being installed in the same place and being
// considered different products. An alternative (better, IMO), is to place
// running number here (like we had with SVN or when we had the gitRev file).
#define BACKEND_VERSION_BUILD 0 //  DO NOT CHANGE THIS NUMBER!!!!


// Double macro expansion taken from here: http://stackoverflow.com/a/5459929/871910
// Lengthy macro names are here to prevent pollution the global namespace with these macro names
#define BACKEND_VERSION_STR_HELPER(x) #x
#define BACKEND_VERSION_STR(x) BACKEND_VERSION_STR_HELPER(x)

#define BACKEND_VERSION BACKEND_VERSION_STR(BACKEND_VERSION_MAJOR) ", " BACKEND_VERSION_STR(BACKEND_VERSION_MINOR) ", " BACKEND_VERSION_STR(BACKEND_VERSION_REVISION) ", " BACKEND_VERSION_STR(BACKEND_VERSION_BUILD)


#endif
