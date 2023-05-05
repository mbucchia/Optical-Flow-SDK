/*
* Copyright 2022 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

#pragma once

#ifndef SECURE_LIBRARY_LOAD_H
#define SECURE_LIBRARY_LOAD_H

#if defined _MSC_VER

#define _UNICODE 1
#define UNICODE 1

#include <windows.h>
#include <wincrypt.h>
#include <wintrust.h>
#include <stdio.h>
#include <direct.h>
#include <tchar.h>
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include <Softpub.h>

#define NV_NVIDIA_CERT_NAME "NVIDIA "

#pragma comment(lib, "crypt32.lib")
#pragma comment (lib, "wintrust")

#define ENCODING (X509_ASN_ENCODING | PKCS_7_ASN_ENCODING)

typedef struct {
    LPWSTR lpszProgramName;
    LPWSTR lpszPublisherLink;
    LPWSTR lpszMoreInfoLink;
} SPROG_PUBLISHERINFO, *PSPROG_PUBLISHERINFO;

/*
* Checks if "NvOFFRUC.dll" is signed with NVIDIA certificate
*/
BOOL isOFNvOFFRUCLibrarySigned(PCCERT_CONTEXT pCertContext)
{
    BOOL fReturn = FALSE;
    LPTSTR szName = NULL;
    DWORD dwData = 0;

    __try
    {
        // Initialize the WINTRUST_FILE_INFO structure.

        WINTRUST_FILE_INFO FileData;
        WINTRUST_DATA WinTrustData;
        GUID WVTPolicyGUID = WINTRUST_ACTION_GENERIC_VERIFY_V2;

        memset(&FileData, 0, sizeof(FileData));
        FileData.cbStruct = sizeof(WINTRUST_FILE_INFO);
        FileData.pcwszFilePath = L"NvOFFRUC.dll";

        memset(&WinTrustData, 0, sizeof(WinTrustData));
        WinTrustData.cbStruct = sizeof(WinTrustData);
        WinTrustData.dwUIChoice = WTD_UI_NONE;
        WinTrustData.fdwRevocationChecks = WTD_REVOKE_WHOLECHAIN;
        WinTrustData.dwUnionChoice = WTD_CHOICE_FILE;
        WinTrustData.dwStateAction = WTD_STATEACTION_IGNORE;
        WinTrustData.dwProvFlags |= WTD_CACHE_ONLY_URL_RETRIEVAL;
        WinTrustData.pFile = &FileData;

        if (WinVerifyTrust(
            NULL,
            &WVTPolicyGUID,
            &WinTrustData) == ERROR_SUCCESS)
        {
            // Allocate memory for subject name.
            dwData = pCertContext->pCertInfo->SerialNumber.cbData;

            // Get Subject name size.
            if (!(dwData = CertGetNameString(pCertContext,
                CERT_NAME_SIMPLE_DISPLAY_TYPE,
                0,
                NULL,
                NULL,
                0)))
            {
                _tprintf(_T("CertGetNameString failed.\n"));
                __leave;
            }

            szName = (LPTSTR)LocalAlloc(LPTR, dwData * sizeof(TCHAR));
            if (!szName)
            {
                _tprintf(_T("Unable to allocate memory for subject name.\n"));
                __leave;
            }

            // Get subject name.
            if (!(CertGetNameString(pCertContext,
                CERT_NAME_SIMPLE_DISPLAY_TYPE,
                0,
                NULL,
                szName,
                dwData)))
            {
                _tprintf(_T("CertGetNameString failed.\n"));
                __leave;
            }
            return (_strnicmp(szName, NV_NVIDIA_CERT_NAME, strlen(NV_NVIDIA_CERT_NAME)) == 0);
        }
        else
        {
            return false;
        }
    }
    __finally
    {
        if (szName != NULL) LocalFree(szName);
    }

    return fReturn;
}

/*
* Allocate string with wchar
*/
LPWSTR AllocateAndCopyWideString(LPCWSTR inputString)
{
    LPWSTR outputString = NULL;

    outputString = (LPWSTR)LocalAlloc(LPTR,
        (wcslen(inputString) + 1) * sizeof(WCHAR));
    if (outputString != NULL)
    {
        lstrcpyW(outputString, inputString);
    }
    return outputString;
}

/*
* Checks if "NvOFFRUC.dll" is signed by NVIDIA, loads the DLL and  returns handle to pointer of HINSTANCE
*/
BOOL SecureLoadLibrary(LPWSTR strLibraryPath, HINSTANCE* hDLL)
{
    WCHAR szFileName[MAX_PATH];
    HCERTSTORE hStore = NULL;
    HCRYPTMSG hMsg = NULL;
    PCCERT_CONTEXT pCertContext = NULL;
    BOOL fResult = FALSE;
    DWORD dwEncoding = 0, dwContentType = 0, dwFormatType = 0;
    PCMSG_SIGNER_INFO pSignerInfo = NULL;
    PCMSG_SIGNER_INFO pCounterSignerInfo = NULL;
    DWORD dwSignerInfo = 0;
    CERT_INFO CertInfo = {0};
    SPROG_PUBLISHERINFO ProgPubInfo = {0};

    ZeroMemory(&ProgPubInfo, sizeof(ProgPubInfo));
    __try
    {
# ifdef UNICODE
        lstrcpynW(szFileName, (strLibraryPath), MAX_PATH);
#else
        if (mbstowcs(szFileName, strLibraryPath, MAX_PATH) == -1)
        {
            printf("Unable to convert to unicode.\n");
            __leave;
        }
#endif

        // Get message handle and store handle from the signed file.
        fResult = CryptQueryObject(CERT_QUERY_OBJECT_FILE,
            szFileName,
            CERT_QUERY_CONTENT_FLAG_PKCS7_SIGNED_EMBED,
            CERT_QUERY_FORMAT_FLAG_BINARY,
            0,
            &dwEncoding,
            &dwContentType,
            &dwFormatType,
            &hStore,
            &hMsg,
            NULL);
        if (!fResult)
        {
            _tprintf(_T("CryptQueryObject failed with %x\n"), GetLastError());
            __leave;
        }

        // Get signer information size.
        fResult = CryptMsgGetParam(hMsg,
            CMSG_SIGNER_INFO_PARAM,
            0,
            NULL,
            &dwSignerInfo);
        if (!fResult)
        {
            _tprintf(_T("CryptMsgGetParam failed with %x\n"), GetLastError());
            __leave;
        }

        // Allocate memory for signer information.
        pSignerInfo = (PCMSG_SIGNER_INFO)LocalAlloc(LPTR, dwSignerInfo);
        if (!pSignerInfo)
        {
            _tprintf(_T("Unable to allocate memory for Signer Info.\n"));
            __leave;
        }

        // Get Signer Information.
        fResult = CryptMsgGetParam(hMsg,
            CMSG_SIGNER_INFO_PARAM,
            0,
            (PVOID)pSignerInfo,
            &dwSignerInfo);
        if (!fResult)
        {
            _tprintf(_T("CryptMsgGetParam failed with %x\n"), GetLastError());
            __leave;
        }

        // Search for the signer certificate in the temporary 
        // certificate store.
        CertInfo.Issuer = pSignerInfo->Issuer;
        CertInfo.SerialNumber = pSignerInfo->SerialNumber;

        pCertContext = CertFindCertificateInStore(hStore,
            ENCODING,
            0,
            CERT_FIND_SUBJECT_CERT,
            (PVOID)& CertInfo,
            NULL);
        if (!pCertContext)
        {
            _tprintf(_T("CertFindCertificateInStore failed with %x\n"),
                GetLastError());
            __leave;
        }

        //Check Signer certificate information.
        if (isOFNvOFFRUCLibrarySigned(pCertContext))
        {
            *hDLL = LoadLibraryExW(L"NvOFFRUC.dll", NULL, NULL);
        }
    }
    __finally
    {
        // Clean up.
        if (ProgPubInfo.lpszProgramName != NULL)
            LocalFree(ProgPubInfo.lpszProgramName);
        if (ProgPubInfo.lpszPublisherLink != NULL)
            LocalFree(ProgPubInfo.lpszPublisherLink);
        if (ProgPubInfo.lpszMoreInfoLink != NULL)
            LocalFree(ProgPubInfo.lpszMoreInfoLink);

        if (pSignerInfo != NULL) LocalFree(pSignerInfo);
        if (pCounterSignerInfo != NULL) LocalFree(pCounterSignerInfo);
        if (pCertContext != NULL) CertFreeCertificateContext(pCertContext);
        if (hStore != NULL) CertCloseStore(hStore, 0);
        if (hMsg != NULL) CryptMsgClose(hMsg);
    }
    return 0;
}

#endif

#endif // SECURE_LIBRARY_LOAD