//! Helper functions for dealing with text data files in
//! a "SIMple SVMlight" (simsvm) format, i.e.,
//! <target> <feature> <feature>...
//! where target and features should be contiguous non-negative integers.



use rayon::iter::ParallelIterator;

use crate::scanner::DelimIter;

/// Given a [`DelimIter`] pointing to the front of a line in a
/// simsvm file, this wrapper is a convenient iterator over
/// just the features in that line.
#[derive(Clone)]
pub struct SimSvmLineIter<'a> {
    target: &'a [u8],
    iter: DelimIter<'a>,
}

pub fn parse(mut iter: DelimIter<'_>) -> SimSvmLineIter<'_> {
    let target = iter.next().expect("target");
    SimSvmLineIter { target, iter }
}

impl<'a> Iterator for SimSvmLineIter<'a> {
    type Item = u32;

    fn next(&mut self) -> Option<u32> {
        self.iter.next().map(|word| {
            let string = std::str::from_utf8(word).expect("utf-8");
            string.parse().expect("parse feature")
        })
    }
}

impl<'a> SimSvmLineIter<'a> {
    pub fn target(&self) -> u32 {
        std::str::from_utf8(self.target)
            .expect("utf-8")
            .parse()
            .expect("parse u32 target")
    }
}
