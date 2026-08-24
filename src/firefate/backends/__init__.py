"""Third-party engines FIREFate drives, each behind its own subpackage.

Every ``import dictys`` (and, as the other two modules land, every celloracle and
SLIDE-R call) lives under here rather than in the domain modules, so the engines
stay swappable and the domain code stays testable against fakes.
"""
