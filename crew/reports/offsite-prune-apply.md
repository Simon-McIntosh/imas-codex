# Offsite registry retention apply

The locked tiered policy was applied once to `ghcr.io/simon-mcintosh/imas-codex-graph`. The fresh dry-run classified all 48 versions as 25 keep and 23 delete; the apply command exited `0` and reported `Pruned 23/23 versions`; the GitHub Packages API then returned exactly the 25 expected survivors. A post-apply dry-run classified those survivors as 25 keep and 0 delete.

## Commands and quantitative gate

| Check | Result |
|---|---|
| GitHub Packages API before count | `48` |
| New recovery point before apply | API id `1199464046`, created `2026-09-02T09:14:57Z`, tag `dev-198ec82-20260902T085827Z-r1` |
| Fresh dry-run | `imas-codex graph prune --registry ghcr.io/simon-mcintosh --dry-run`; exit `0`; `25 keep`, `23 delete`; no changes made |
| Authorized apply | `imas-codex graph prune --registry ghcr.io/simon-mcintosh --force`; exit `0`; `23/23` deleted, `0` failed |
| GitHub Packages API after count | `25` |
| Post-apply policy scan | exit `0`; `25 keep`, `0 delete`; `Nothing to prune under tiered retention` |

The before/after cardinality delta is `48 - 25 = 23`, equal to both the fresh dry-run delete count and the apply success count. The API survivor set is exactly the dry-run keep set, so the deleted set is exactly the dry-run delete set rather than merely the same size.

## Kept versions

Every dry-run keep row survived in the GitHub Packages API:

| Tier | Created (UTC) | API id | Tags |
|---|---:|---:|---|
| weekly | 2026-09-02T09:14:57Z | 1199464046 | `dev-198ec82-20260902T085827Z-r1` |
| latest | 2026-07-07T12:27:23Z | 1008109688 | `v5.3.0-rc6`, `latest` |
| release | 2026-06-30T07:48:18Z | 987002247 | `v5.3.0-rc5` |
| release | 2026-04-17T16:35:13Z | 804521518 | `v5.3.0-rc4` |
| release | 2026-04-10T13:25:55Z | 789252927 | `v5.3.0-rc2` |
| release | 2026-04-10T07:35:22Z | 788570634 | `v5.3.0-rc1` |
| release | 2026-04-09T21:20:04Z | 787825882 | `v5.2.0-rc10` |
| release | 2026-04-09T17:25:39Z | 787406885 | `v5.2.0-rc8` |
| release | 2026-04-09T12:28:06Z | 786696661 | `v5.2.0-rc1` |
| release | 2026-04-09T10:07:57Z | 786379386 | `v5.2.0-rc2` |
| release | 2026-04-08T16:14:43Z | 784726084 | `v5.1.0-rc16` |
| release | 2026-04-08T14:12:30Z | 784435645 | `v5.1.0-rc15` |
| release | 2026-04-08T13:55:05Z | 784392577 | `v5.1.0-rc14` |
| weekly | 2026-04-08T13:40:34Z | 784354558 | `5.1.0rc13.dev0-ge16c5c8db.d20260408-r1` |
| release | 2026-04-08T11:37:59Z | 784080033 | `v5.1.0-rc12` |
| release | 2026-04-08T07:51:04Z | 783613738 | `v5.1.0-rc10` |
| release | 2026-03-27T09:48:27Z | 761420845 | `v5.0.0-rc11` |
| release | 2026-03-27T08:29:14Z | 761293051 | `v5.0.0-rc10` |
| release | 2026-03-26T18:28:25Z | 760252081 | `v5.0.0-rc9` |
| release | 2026-03-26T16:03:53Z | 759955468 | `v5.0.0-rc8` |
| weekly | 2026-03-26T15:31:55Z | 759873333 | `v5.0.0-fix-test` |
| weekly | 2026-03-26T15:14:23Z | 759829554 | `v-direct-test` |
| monthly | 2026-03-18T12:13:01Z | 743415309 | `4.0.1.dev1849-gde0bd2cb5.d20260316-r1` |
| monthly | 2026-02-26T17:51:54Z | 707500702 | `4.0.1.dev1111-g1f1cc9918.d20260226-r1` |
| monthly | 2026-01-30T12:53:29Z | 664115721 | `3.2.1.dev588-g2d31208df.d20260129` |

Protection proof: all 17 release-tier rows survived; the version carrying `latest` survived; and the new weekly recovery point survived. The two legacy tags containing “test” but not beginning with `test-` remain correctly classified as weekly under the locked policy.

## Deleted versions

The apply command reported success for every fresh dry-run delete row:

| Tier | Created (UTC) | Version identity |
|---|---:|---|
| delete-untagged | 2026-04-09T12:15:53Z | `untagged@786668925` (`sha256:2f9c19d0d702b66115fc9c6513bd2f2138b8b0e5385bb56a18b4b32983519664`) |
| delete-untagged | 2026-04-09T12:04:27Z | `untagged@786641564` (`sha256:0c566dcc432226368cca8064d322c70b548125b49486a7c51ea220dd42d8949e`) |
| delete-untagged | 2026-04-09T09:58:24Z | `untagged@786358004` (`sha256:d9b2033a3f73a9be59b40a2cbddec04c8b27c43bf0df65756c4950ce1772fea6`) |
| delete-untagged | 2026-04-08T15:47:45Z | `untagged@784663303` (`sha256:4bdd95bf53cdc1c1b6a10021033ebfc1e28f43d31720ac605873ef7b1b7bb5da`) |
| delete-untagged | 2026-04-08T14:50:23Z | `untagged@784529717` (`sha256:02df02e62f37cf9eaeb380f3bb65a753438c0b219f299faf7a570226936c1bd4`) |
| delete-untagged | 2026-03-27T09:14:41Z | `untagged@761363312` (`sha256:b203a25ab01a3a04304150fd5e6547e2bc7c8e3eb1c92107e9d349c2f3042081`) |
| delete-untagged | 2026-03-26T18:16:59Z | `untagged@760231674` (`sha256:81f04741a0003dab3ce776ab358458b12bfc389c56eb4e8c3b696be8816f66ff`) |
| delete-untagged | 2026-03-26T15:39:28Z | `untagged@759894146` (`sha256:46ae5a939867f6b1842b9fd4dd15098af954f7ce35356db9633dd915f34ec8e8`) |
| delete-test | 2026-03-26T15:02:14Z | `test-manual-push` |
| delete-test | 2026-03-26T14:56:10Z | `test-push` |
| delete-thinned | 2026-03-16T05:28:30Z | `4.0.1.dev1815-g500947550-r1` |
| delete-thinned | 2026-03-09T11:07:55Z | `4.0.1.dev1459-gbcee2a9b3-r1` |
| delete-thinned | 2026-03-06T17:54:51Z | `4.0.1.dev1400-gb3e7d922e.d20260305-r3` |
| delete-thinned | 2026-03-06T17:35:11Z | `4.0.1.dev1400-gb3e7d922e.d20260305-r2` |
| delete-thinned | 2026-03-06T15:07:55Z | `4.0.1.dev1400-gb3e7d922e.d20260305-r1` |
| delete-thinned | 2026-02-25T20:47:43Z | `4.0.1.dev1055-ga1801fe76-r1` |
| delete-thinned | 2026-02-20T12:44:41Z | `3.2.1.dev1262-g8f358e01d.d20260220-r2` |
| delete-thinned | 2026-02-20T12:36:56Z | `3.2.1.dev1262-g8f358e01d.d20260220-r1` |
| delete-thinned | 2026-02-17T20:11:20Z | `3.2.1.dev1142-g993f1fea3.d20260217-r1` |
| delete-thinned | 2026-02-09T05:35:06Z | `3.2.1.dev842-gbd3c0f96c.d20260205` |
| delete-untagged | 2026-02-06T12:40:56Z | `untagged@674327083` (`sha256:a94015f4b8f615d683567816a96bb186457e235f819dc4880be9b9c4f060f22e`) |
| delete-thinned | 2026-02-03T08:56:38Z | `3.2.1.dev684-g3659e7fbf` |
| delete-untagged | 2026-02-03T08:51:51Z | `untagged@668545896` (`sha256:5cb8b40a7908e4e0a9056c6abbcc4881803d43a2dd788a8358be1edfe2229dac`) |

Deleted-tier totals were 10 untagged, 2 test-prefixed, and 11 thinned versions. No release, `latest`, weekly, or monthly row was deleted.
