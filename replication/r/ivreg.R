# ivrobust replication placeholder
# Requires AER package.

library(AER)

fixture <- read.csv("../data/weak_iv_fixture.csv")

# Example IV regression
fit <- ivreg(y ~ d + x1 | z1 + z2 + z3 + x1, data = fixture)
summary(fit, vcov = sandwich::vcovHC)

# Save outputs manually for golden tables.
