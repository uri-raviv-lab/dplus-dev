# Troubleshoots
This file is a collection of problems that rose up when trying to compile D+ and how these were fixed.

## 1. ALINK operation failed (80040436) : Error signing assembly -- Access is denied.
This problem was caused by, as is written, a lack of access permissions.
It is fixed by going to C:\Users\All Users\Application Data\Microsoft\Crypto\RSA\MachineKeys and right click this folder, select Properties-Security and assign ownership to yourself and give yourself full control.

This fixed was found here: https://developercommunity.visualstudio.com/t/al1078-error-signing-assembly-access-denied/887321
