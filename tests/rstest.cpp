/* Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <algorithm>
#include <rangeset.hpp>

using namespace std;

std::ostream &operator <<(std::ostream &o, std::pair<const int, int> &r) {
    o << "[" << r.first << "," << r.second << "]";
    return o;
}

std::ostream &operator << (std::ostream &o, RangeSet &rs) {
#if __cplusplus <= 199711L // not C++-11, hopefully C++-98
    for (RangeSet::iterator _s = rs.begin(); _s != rs.end(); ++_s) { Range s = *_s;
#else
    for (auto s: rs) {
#endif
        std::cout << s << std::endl;
    }
    return o;
}

int main()
{
    RangeSet rs;

    Range r1(1,2);
    Range r2(4,5);
    Range r3(8,22);

    rs.insert(r1);
    rs.insert(r2);
    rs.insert(r3);

    cout << "rs contains:" << endl;
    cout << rs;
    cout << endl;

    { RangeSet::find_result res = rs.find(r1); Range r = *res.first; assert(r == r1); assert(res.second == RangeSet::fully_contained); }
    
    { RangeSet::find_result res = rs.find(r2); Range r = *res.first; assert(r == r2); assert(res.second == RangeSet::fully_contained); }

    { 
        RangeSet::find_result res = rs.find(r3); 
        Range r = *res.first; 
        if (r.second == RangeSet::not_found) {
            assert(res.first == rs.end());
            cout << "cannot find range " << r3 << endl; 
        }
        else {
            if (r != r3) cout << "ERROR: " << r3 << endl;
            if (res.second != RangeSet::fully_contained) cout << "ERROR: result " << res.second << endl;
        }
    }
    
    { Range r(8,9); RangeSet::find_result res = rs.find(r); Range ro = *res.first; cout << r << " is contained in " << ro << endl; assert(res.second == RangeSet::fully_contained); }

    { Range r(7,9); RangeSet::find_result res = rs.find(r); Range ro = *res.first; cout << r << " partially overlaps with " << ro << endl; assert(res.second == RangeSet::partial_overlap); }

    {
        Range r4(5,6);
        cout << "range extension test" << endl;
        cout << "adding " << r4 << endl;
        RangeSet::insert_result res = rs.insert(r4);
        cout << "res.first=" << *res.first << " res.second=" << res.second << endl;
        assert(*res.first == Range(4,6));
        assert(res.second);
    }

    cout << "rs contains:" << endl;
    cout << rs;
    cout << endl;

    {
        Range r5(5,9);
        cout << "range extension test" << endl;
        cout << "adding " << r5 << endl;
        RangeSet::insert_result res = rs.insert(r5);
        cout << "res.first=" << *res.first << " res.second=" << res.second << endl;
        //assert(*res.first == Range(4,6));
        //assert(res.second);
    }

    cout << "rs contains:" << endl;
    cout << rs;
    cout << endl;

}
