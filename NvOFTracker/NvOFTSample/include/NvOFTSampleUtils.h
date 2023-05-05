/*
* Copyright 2019-2022 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

#pragma once
#include <sstream>
#include <iostream>
#include <string>
#include <ctype.h>

struct CommandLineFields
{
    std::string inputFile;
    std::string outputFile;
    std::string detectorEngineFile;
    std::string trackerDumpFile;
    uint32_t skipInterval = 0;
    uint32_t gpuId = 0;
    bool dumpToConsole = false;
};

void ShowHelpAndExit(const char* szBadOption, const char* szOptionalError = nullptr)
{
    std::ostringstream oss;
    bool bThrowError = false;
    if (szBadOption)
    {
        bThrowError = true;
        oss << "Error parsing \"" << szBadOption << "\"" << std::endl;
    }
    if (szOptionalError)
    {
        bThrowError = true;
        oss << szOptionalError << std::endl;
    }
    oss << "Command Line Options:" << std::endl
        << "Mandatory Parameters: " << std::endl
        << "-i           Input video file" << std::endl
        << "-e           TensorRT Engine file for Detector" << std::endl
        << "Optional Parameters: " << std::endl
        << "-o           Output video file" << std::endl
        << "-fT          Filename to dump tracked objects" << std::endl
        << "-dC          Dump tracked objects to Console" << std::endl
        << "-sI          Detection skip interval. 0 <= sI <= 4" << std::endl
        << "-g           GPU Id on which the tracker needs to run. Default is 0" << std::endl
        ;
    if (bThrowError)
    {
        throw std::invalid_argument(oss.str());
    }
    else
    {
        std::cout << oss.str();
        exit(0);
    }
}

void CheckMandatoryParams(const CommandLineFields& clFields)
{
    if (clFields.inputFile.empty())
    {
        ShowHelpAndExit(nullptr, "Input file not specified");
    }
    if (clFields.detectorEngineFile.empty())
    {
        ShowHelpAndExit(nullptr, "Detector file not specified");
    }
}

void ParseCommandLine(int argc, char *argv[], CommandLineFields& clFields)
{
    std::ostringstream oss;
    int i;
    auto IsEqual = [](const std::string& a, const std::string& b) -> bool
    {
        auto size = a.size();
        if (b.size() != size)
        {
            return false;
        }
        for (int j = 0; j < size; ++j)
        {
            if (tolower(a[j]) != tolower(b[j]))
            {
                return false;
            }
        }
        return true;
    };

    for (i = 1; i < argc; i++) {
        if (IsEqual(argv[i], "-h")) {
            ShowHelpAndExit(nullptr);
        }
        if (IsEqual(argv[i], "-i")) {
            if (++i == argc && clFields.inputFile.empty()) {
                ShowHelpAndExit("-i");
            }
            clFields.inputFile.assign(argv[i]);
            continue;
        }
        if (IsEqual(argv[i], "-e")) {
            if (++i == argc && clFields.detectorEngineFile.empty()) {
                ShowHelpAndExit("-e");
            }
            clFields.detectorEngineFile.assign(argv[i]);
            continue;
        }
        if (IsEqual(argv[i], "-o")) {
           if (++i == argc && clFields.outputFile.empty()) {
               ShowHelpAndExit("-o");
            }
            clFields.outputFile.assign(argv[i]);
            continue;
        }
        if (IsEqual(argv[i], "-fT")) {
            if (++i == argc && clFields.trackerDumpFile.empty()) {
                ShowHelpAndExit("-fT");
            }
            clFields.trackerDumpFile.assign(argv[i]);
            continue;
        }
        if (IsEqual(argv[i], "-sI")) {
           if (++i == argc) {
               ShowHelpAndExit("-sI");
            }
            auto val = std::stoi(std::string(argv[i]));
            if (val < 0)
            {
                ShowHelpAndExit("-sI");
            }
            clFields.skipInterval = (uint32_t)val;
            continue;
        }
        if (IsEqual(argv[i], "-g")) {
            if (++i == argc) {
                ShowHelpAndExit("-g");
            }
            auto val = std::stoi(std::string(argv[i]));
            if (val < 0)
            {
                ShowHelpAndExit("-g");
            }
            clFields.gpuId = val;
            continue;
        }
        if (IsEqual(argv[i], "-dC")) {
            clFields.dumpToConsole = true;
            continue;
        }
        ShowHelpAndExit(argv[i]);
    }
    CheckMandatoryParams(clFields);
}
