X+ v1.0 README

Tal Ben-Nun, Pablo Szekely, Avi Ginsburg and Uri Raviv

System Requirements:
~~~~~~~~~~~~~~~~~~~~
This program uses Microsoft .NET Framework 3.0 and Microsoft Visual C++ 2008 Redistributable 
Package, both available for download on the Microsoft website.

Microsoft .NET Framework 3.0 is available for download at this URL:
http://www.microsoft.com/downloads/details.aspx?FamilyID=10CC340B-F857-4A14-83F5-25634C3BF043&displaylang=en

Microsoft Visual C++ 2008 Redistributable Package is available for download at this URL:
http://www.microsoft.com/downloads/details.aspx?familyid=A5C84275-3B97-4AB7-A40D-3802B2AF5FC2&displaylang=en

Downloading X+:
~~~~~~~~~~~~~~~
X+ can be found in the following link:
http://chemistry.huji.ac.il/~raviv/xplus.zip

Installing X+:
~~~~~~~~~~~~~~

Ensure that .NET and Visual C++ 2008 Redistributable Package are installed.  Copy X+.exe and xplusbackend.dll to the desired directory and run X+.

Installing X+ Under Linux:
~~~~~~~~~~~~~~~~~~~~~~~~~~

X+ can be run under Linux using Wine. The following steps must be taken before running X+ for the first time:

# apt-get install wine

$ wget http://winezeug.googlecode.com/svn/trunk/winetricks
$ sh winetricks corefonts dotnet20
$ sh winetricks vcrun2008
$ sh winetricks gdiplus

Run X+:
$ wine X+.exe



Usage:
~~~~~~

See the Quick Users Guide for more information.
