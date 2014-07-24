/*
  mallocMC: Memory Allocator for Many Core Architectures.
  https://www.hzdr.de/crp

  Copyright 2014 Institute of Radiation Physics,
                 Helmholtz-Zentrum Dresden - Rossendorf

  Author(s):  Carlchristian Eckert - c.eckert ( at ) hzdr.de

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
  THE SOFTWARE.
*/

#pragma once

#include <iostream>
#include <sstream>
#include <vector>
#include <utility>


// prints a variable in combination with it's name
// source: http://www.cplusplus.com/forum/beginner/11252/
#define TO_STREAM(stream,variable) (stream) <<#variable": "<<(variable) 
#define NAME_TO_STRING(variable) std::string(#variable)


template <typename T>
std::string t2s(T in){
  std::stringstream ss;
  ss << in;
  return ss.str();
}

#define MK_STRINGPAIR(variable) std::make_pair(std::string(#variable),t2s(variable))

/**
 * prints all parameters machine readable
 *
 * for params, see run_heap_verification-internal parameters
 */
void print_machine_readable(std::vector<std::pair<std::string, std::string> > output){

  std::string sep = ",";
  std::stringstream h;
  std::stringstream v;

  for(int i=0; i<output.size(); ++i){
    h << output[i].first;
    if(i+1 < output.size()) h << sep;

    v << output[i].second;
    if(i+1 < output.size()) v << sep;
  }

  std::cout << h.str() << std::endl;
  std::cout << v.str() << std::endl;
}
