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
#include <cstring>
#include <sstream>
#include <vector>
#include <map>
#include <assert.h>
#include <stdio.h>
#include <typeinfo>
#include <cctype>
#include <type_traits>
#include <algorithm>
#include <functional>
#include <iostream>
#include <stdint.h>

#define NVOFFRUC_ARGS_PARSE(parser, argc, argv)          \
    try {                                            \
        (parser).Parse((argc), (argv));              \
    } catch(const std::exception &e) {               \
        std::cerr << e.what() << std::endl;          \
         exit(1);                                    \
    }

template <typename T>
std::string ToString(T val)
{
    std::ostringstream stream;
    stream << val;
    return stream.str();
}

template <>
inline std::string ToString(bool bVal)
{
    std::ostringstream stream;
    stream << std::boolalpha << bVal;
    return stream.str();
}

template <>
inline std::string ToString(std::string strVal)
{
    std::ostringstream stream;
    stream << (strVal.empty() ? "None" : strVal);
    return stream.str();
}

inline bool ParseOption(
        const std::string& strArg,
        const std::string& strOption,
        const std::function<bool(const std::string&)>& parseFunc,
        bool bRequiresValue,
        bool* bHasValue   )
{
    bool hasSingleDash = strArg.length() > 1 && strArg[0] == '-';
    *bHasValue = false;
    if (hasSingleDash)
    {
        bool hasDoubleDash = strArg.length() > 2 && strArg[1] == '-';
        std::string strKey = strArg.substr(hasDoubleDash ? 2 : 1);
        int val = 1;
        std::string value = std::to_string(val);
        size_t equalsPos = strKey.find('=');
        if (equalsPos != std::string::npos)
        {
            value = strKey.substr(equalsPos + 1);
            strKey = strKey.substr(0, equalsPos);
        }
        else if (bRequiresValue)
        {
            return false;
        }
        if (strKey != strOption)
        {
            return false;
        }
        *bHasValue = parseFunc(value);
        return true;
    }
    return false;
}

template <typename T>
bool ParseOption(
        const std::string& optValue, 
        T& optVal)
{
    std::istringstream stream(optValue);
    stream >> optVal;
    if (!stream.eof() && !stream.good()) 
    {
        return false;
    }
    return true;
}

template <>
inline bool ParseOption(
                const std::string& strOptValue, 
                bool& bOptVal)
{
    std::string temp = strOptValue;
    std::transform(temp.begin(), temp.end(), temp.begin(),
        [](char c) -> char { return (char)(std::tolower(c)); });
    std::istringstream is(temp);
    is >> (temp.size() > 1 ? std::boolalpha : std::noboolalpha) >> bOptVal;
    return true;
}

template <typename T>
std::string GetTypeName(T val)
{
    if (std::is_same<T, int32_t>::value)
    {
        return "int32";
    }
    else if (std::is_same<T, int64_t>::value)
    {
        return "int64";
    }
    else if (std::is_same<T, float>::value)
    {
        return "float";
    }
    else if (std::is_same<T, std::string>::value)
    {
        return "string";
    }
    else if (std::is_same<T, bool>::value)
    {
        return "bool";
    }
    return "unknown";
}

/*
 * The Option class represents an option (or command-line parameter) which
 * can accept at most one value. If the option requires a value, the type of
 * that values can be one of int32_t, int64_t, float, std::string and bool.
 */
class Option
{
public:
    template <typename T>
    static Option CreateOption(
                    const char* pName,
                    const char* pUsage,
                    T& val,
                    bool bRequiresValue = true)
    {
        auto callbackFunc = [&val](const std::string& optValue) ->bool
        {
            return ParseOption(optValue, val);
        };

        auto printFunc = [&val]() -> std::string
        {
            return ToString(val);
        };

        return Option(pName, callbackFunc, pUsage,
            printFunc, GetTypeName(val), ToString(val), bRequiresValue);
    }
private:

    Option(
        const char* name,
        std::function<bool(const std::string&)> callback,
        const char* desc,
        std::function<std::string()> printFunc,
        const std::string& typeName,
        const std::string& defaultValue,
        bool bRequiresValue)
        : m_name(name),
        m_callback(callback),
        m_getOptValueForDisplay(printFunc),
        m_usageDesc(desc),
        m_typeName(typeName),
        m_defaultValueForDisplay(defaultValue),
        m_bRequiresValue(bRequiresValue)
    {

    }

    bool Parse(
            const std::string& arg,
            bool* bHasValue) const
    {
        return ParseOption(arg, m_name, m_callback, m_bRequiresValue, bHasValue);
    }

    std::string m_name;
    std::string m_defaultValueForDisplay;
    std::function<bool(const std::string&)> m_callback;
    std::function<std::string()> m_getOptValueForDisplay;

    std::string m_usageDesc;
    std::string m_typeName;
    bool m_bRequiresValue;
    friend class CmdParser;
};

/*
 * The CmdParser class represents a parser for command-line options.
 */
class CmdParser
{
public:

    CmdParser()
    {

    }
    template <typename T>
    void AddOptions(
            const char* szName,
            T& value,
            const char* szDesc)
    {
        m_options.emplace_back(Option::CreateOption(szName, szDesc, value));
        auto option = m_options.back();
        std::ostringstream oss;
        oss << "--" << option.m_name << "=<" << option.m_typeName << ">";
        m_maxArgWidth = m_maxArgWidth > oss.str().size() ? m_maxArgWidth : oss.str().size();
    }

    bool Parse(
            int argc,
            const char** argv)
    {
        bool bResult = true;
        if (argc == 1 || std::strcmp(argv[1], "--help") == 0)
        {
            std::cout << help(argv[0]) << std::endl;
            exit(0);
        }
        for (int i = 1; i < argc; ++i)
        {
            if (std::strcmp(argv[i], "--help") != 0)
            {
                bool bFound = false;
                for (const auto& option : m_options)
                {
                    bool bHasValue;
                    bFound = option.Parse(argv[i], &bHasValue);
                    if (!bHasValue)
                    {
                        bResult = false;
                    }
                    if (bFound)
                    {
                        break;
                    }
                }
                if (!bFound)
                {
                    std::ostringstream oss;
                    oss << "Error parsing \"" << argv[i] << "\"" << std::endl;
                    oss << help(argv[0]);
                    throw std::invalid_argument(oss.str());
                }
            }
        }
        return bResult && (argc < 2 || std::strcmp(argv[1], "--help") != 0);
    }

    std::string help(const std::string& strCmdline)
    {
        const uint32_t maxTypeNameSize = 10;
        std::string appName;
        size_t pos_s = strCmdline.find_last_of("/\\");
        if (pos_s == std::string::npos)
        {
            appName = strCmdline;
        }
        else
        {
            appName = strCmdline.substr(pos_s + 1, strCmdline.length() - pos_s);
        }
        std::stringstream ss;
        ss << "Usage: " << appName.c_str() << " [params] " << std::endl;
        for (const auto& option : m_options)
        {
            std::stringstream opt;
            auto token = std::string("--");
            opt << token << option.m_name << "=<" << option.m_typeName << ">";
            auto curWidth = opt.str().size();
            ss << opt.str() << std::string(m_maxArgWidth - curWidth, ' ') << "\t"
                << option.m_usageDesc << std::endl;
            ss << std::string(m_maxArgWidth, ' ') << "\t"
                << "Default value: " << option.m_defaultValueForDisplay << std::endl;
        }

        return ss.str();
    }
private:
    std::vector<Option> m_options;
    size_t m_maxArgWidth = 0;
};
#pragma once

