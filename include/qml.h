// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Korea Platform Service Technology Co., Inc. All rights reserved.

#ifndef QML_H
#define QML_H

// QML Core Header
// qml = query-based machine learning: session engine and parser

#include <string>
#include <vector>
#include <memory>

namespace qml {

// QML Parser Class
class Parser {
public:
    // constructs an empty parser
    //  in: (none)
    // out: (constructor)
    Parser();
    // destroys the parser
    //  in: (none)
    // out: (destructor)
    ~Parser();
    
    // parses QML script text into its constituent elements
    //  in: code — QML script text
    // out: parsed elements in source order (declared interface; no in-tree
    //      implementation yet)
    std::vector<std::string> parse(const std::string& code);
    // syntax-checks QML script text without executing it
    //  in: code — QML script text
    // out: true if the script is well-formed (declared interface; no in-tree
    //      implementation yet)
    bool validate(const std::string& code);
};

// QML Engine Interface
class Engine {
public:
    // virtual destructor so concrete engines destroy correctly through
    // an Engine pointer
    //  in: (none)
    // out: (destructor)
    virtual ~Engine() = default;
    // runs one QML script against the engine
    //  in: code — QML script text
    // out: none (void; pure virtual — behaviour defined by the implementation)
    virtual void execute(const std::string& code) = 0;
    // loads the model the engine will execute against
    //  in: modelPath — model checkpoint path
    // out: none (void; pure virtual — behaviour defined by the implementation)
    virtual void load_model(const std::string& modelPath) = 0;
};

// Factory methods

// heap-allocates a Parser
//  in: (none)
// out: owning unique_ptr to a new Parser (declared interface; no in-tree
//      implementation yet)
std::unique_ptr<Parser> create_parser();
// heap-allocates a concrete Engine
//  in: (none)
// out: owning unique_ptr to a new Engine (declared interface; no in-tree
//      implementation yet)
std::unique_ptr<Engine> create_engine();

} // namespace qml

#endif // QML_H