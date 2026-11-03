# Scope

User explicitly requests autonomous diagnosis and repair for stuck/erroring Qwen. Preserve original generation inputs/runtime/prompts/schema/reviews and same model/route/credentials. Only operations policy/guard/recovery authorization change.

83regressiontests passed. Unknown403cooldown/revalidation/full8of8gates, permanentdenial/401/userSTOP cases, failedprobes no restart, no permanentunknown403latch, directrecovery authorization and historicalpreservation verified. v5operationalpolicy bound; originalgenerationruntime and currentlyhealthyworker untouched.

Final86tests pass. Recoveryhelper now also requires a real current guardattempt, validates rollingbudget/cooldown, and denies directautoflagbypass. Currenthealthyworker28785unchanged. No actual403injected or synthetic productionrecovery claimed; codepaths verifiedinregressiontests. Automation saved andschedule/status/target preserved.
