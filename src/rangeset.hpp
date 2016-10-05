#pragma once

#include <utility> //for pair
#include <map>
#include <assert.h>
#include <iostream>

#ifndef DBG
//#define DBG(A) A
#define DBG(A) do { } while(0)
#endif

typedef unsigned long Value;

// inclusive range [start,end]
typedef std::pair<const Value, Value> Range;

std::ostream& operator <<(std::ostream &o, const Range &r) {
        o << "[" << std::hex << r.first << "," << r.second << "]" << std::dec;
        return o;
}

#if __cplusplus <= 199711L // not C++-11, hopefully C++-98
namespace std {
        template<class BidirIt>
        BidirIt prev(BidirIt it, typename std::iterator_traits<BidirIt>::difference_type n = 1)
        {
                std::advance(it, -n);
                return it;
        }

        template<class BidirIt>
        BidirIt next(BidirIt it, typename std::iterator_traits<BidirIt>::difference_type n = 1)
        {
                std::advance(it, n);
                return it;
        }
}
#endif

// based on http://stackoverflow.com/questions/32869247/a-container-for-integer-intervals-such-as-rangeset-for-c

class RangeSet : public std::map<Value, Value> {
public:
        typedef std::map<Value, Value> Super;

        typedef void (*callback)(iterator &i);
        callback _pre_erase;

        RangeSet() : _pre_erase(0) {}
        ~RangeSet() {}

        enum embedding {
                not_found = 0,
                partial_overlap,
                fully_contained,
        };
        
        typedef std::pair<iterator, embedding> find_result;

        /*
          a) no entries after range.first
          map:     ---  --------- -----
          range:                     ------

          b) previous entry.second is before range.first
          map:     ---  ---------     -----
                        prev          after
          range:                  ------

          b0) previous entry.second is before range.first
          map:                        -----
                        prev==end     after
          range:                  ------

          c) wholly contained, return original range + false
          map:     ---  ---------     -----
                        insert_range  after
          range:          ------

          d) overlapping, extend insert_range to cover range
          map:     ---  ---------     -----
                        insert_range  after
          range:          --------

        */
        
        find_result find(const Range& range) {

                DBG((std::cout << "size = " << size() 
                     << " range = " << range
                     << std::endl));

                assert(range.first < range.second);

                if (!size())
                        return find_result(end(), not_found);

                iterator after = upper_bound(range.first);
                iterator insert_range;
                DBG((std::cout << "after = " << *after << std::endl));

                if (after != end()) {
                        // b, at least one range following
                        if (range.second > after->first) {
                                return find_result(after, partial_overlap);
                        }
                }
                iterator prev = std::prev(after);
                if (prev == end()) {
                        // empty ?
                        //assert(begin() == end());
                        return find_result(end(), not_found);
                }
                if (prev == after) {
                        // b0, container has a single element
                        return find_result(end(), not_found);
                }
                DBG((std::cout << "prev = " << *prev << std::endl));
                assert(prev->first <= range.first);
                if (prev->second < range.first)
                        return find_result(end(), not_found);

                if (prev->second < range.second)
                        return find_result(prev, partial_overlap);
                else
                        return find_result(prev, fully_contained);
        }

        typedef std::pair<RangeSet::iterator, bool> insert_result;

        insert_result insert(const Range& range) {
                assert(range.first <= range.second);

                DBG((std::cout << "size = " << size() << std::endl));

                RangeSet::iterator insert_range;
                RangeSet::iterator after = upper_bound(range.first);
                if(after == end()) {
                        // a
                        DBG((std::cout << "a) inserting " << range << std::endl));
                        insert_range = Super::insert(after, range);
                }
                else {
                        DBG((std::cout << "after is " << *after << std::endl));
                        // after.first is the 1st strictly greater than range.first
                        assert(after->first > range.first);

                        if (begin() == after ) {
                                // b0, container has 1+ elements following
                                DBG((std::cout << "b0) following elements only, inserting" 
                                     << range << " prior to " << *after << std::endl));
                                insert_range = Super::insert(after, range);
                        }
                        else {
                                insert_range = std::prev(after);
                                if (insert_range->second < range.first) {
                                        DBG((std::cout << "prev  is " << *insert_range << std::endl));
                                        assert(insert_range->first <= range.first);
                                        // b
                                        DBG((std::cout << "b) inserting " << range << " prior to " 
                                             << *after << std::endl));
                                        insert_range = Super::insert(after, range);
                                }
                                else {
                                        if(insert_range->second >= range.second) {
                                                // c
                                                // insert_range     range
                                                // 2aaaaaa b0000    2aaaaaa ab000
                                                // 2aaaaaa b7fff >= 2aaaaaa abfff
                                                DBG((std::cout << range << "c) contained in " << *insert_range << std::endl));
                                                return insert_result(insert_range, false);
                                        }   
                                        else {
                                                // d
                                                DBG((std::cout << "d) extending " << *insert_range << " with " << range << std::endl));
                                                insert_range->second = range.second;
                                        }
                                }
                        }
                }
                // remove ranges overlapping insert_range
                while(after != end() and range.second >= after->first) {
                        DBG((std::cout << "warn: merging " << *after << " with " << *insert_range << std::endl));
                        insert_range->second = std::max(after->second, insert_range->second);
                        if (_pre_erase) _pre_erase(after);
                        after = erase(after); // returns range which follows after
                }   

                DBG((dump()));

                return insert_result(insert_range, true);
        }   

        void dump() {
                iterator i;
                for (i=begin(); i != end(); ++i)
                        std::cout << *i << std::endl;
        }

#if __cplusplus <= 199711L // not C++-11, hopefully C++-98
        iterator erase(iterator pos) {
                iterator tmp = std::next(pos);
                Super::erase(pos);
                return tmp;
        }
#endif

};
