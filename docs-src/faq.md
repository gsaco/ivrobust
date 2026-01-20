# FAQ

## Why is my confidence set disjoint or unbounded?

This can occur under weak identification. In weak-IV robust inference, it is
common that the identified set is wide or even effectively unbounded for
plausible confidence levels.

## Should I report 2SLS standard errors?

You can report them as conventional strong-ID standard errors, but they are not
weak-IV robust. If instrument strength is questionable, report weak-IV robust
inference (for example, an AR confidence set) alongside 2SLS estimates.
