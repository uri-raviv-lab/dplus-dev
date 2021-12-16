//
// This files contains the version of the frontend (DLLs and D+).
//
// It will be updated manually and replaces the old gitrev.h.
//

#ifndef FRONTEND_VERSION_H
#define FRONTEND_VERSION_H

// The version is used in RC files for resources, so it must be broken into elements
#define FRONTEND_VERSION_MAJOR 4 //releases to public, eg each version that goes to a reviewer
#define FRONTEND_VERSION_MINOR 6 //significant new features or changes in how running D+ works
#define FRONTEND_VERSION_REVISION 0//minor changes and bug fixes
// Don't use this field for creating different installer versions as
// WiX (and MSI) MSI specifically ignore this field when comparing versions.
// This leads to many versions being installed in the same place and being
// considered different products. An alternative (better, IMO), is to place
// running number here (like we had with SVN or when we had the gitRev file).
#define FRONTEND_VERSION_BUILD 0 // DO NOT CHANGE THIS NUMBER!!!! very minor changes (only used if needed for some reason)


// Double macro expansion taken from here: http://stackoverflow.com/a/5459929/871910
// Lengthy macro names are here to prevent pollution the global namespace with these macro names
#define FRONTEND_VERSION_STR_HELPER(x) #x
#define FRONTEND_VERSION_STR(x) FRONTEND_VERSION_STR_HELPER(x)

#define FRONTEND_VERSION FRONTEND_VERSION_STR(FRONTEND_VERSION_MAJOR) ", " FRONTEND_VERSION_STR(FRONTEND_VERSION_MINOR) ", " FRONTEND_VERSION_STR(FRONTEND_VERSION_REVISION) ", " FRONTEND_VERSION_STR(FRONTEND_VERSION_BUILD)

#define FRONTEND_VERSION_DECIMAL FRONTEND_VERSION_STR(FRONTEND_VERSION_MAJOR) "." FRONTEND_VERSION_STR(FRONTEND_VERSION_MINOR) "." FRONTEND_VERSION_STR(FRONTEND_VERSION_REVISION) "." FRONTEND_VERSION_STR(FRONTEND_VERSION_BUILD)

#endif
