/* SPDX-License-Identifier: GPL-3.0-or-later */
#include "llm/s2_controller.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
static unsigned failures;
#define CHECK(c) do { if (!(c)) { fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#c); failures++; } } while (0)
static struct janas_s2_observation observation(unsigned kind, unsigned reuse)
{
    struct janas_s2_observation o = {0};
    o.cycle=10; o.warm=1; o.reuse=reuse; o.width=2; o.delivered=2;
    o.full_seconds=1; o.verify_seconds=.8; o.draft_seconds=.1;
    o.expert_cpu_seconds=.5; o.expert_groups=100; o.expert_assignments=140;
    o.root_offered=kind==0; o.rank2matched=kind==0; o.rank2_probability=.6;
    o.continuation_kind=kind;
    return o;
}
static struct janas_s2_offer offer(unsigned reuse)
{
    struct janas_s2_offer o = {0};
    o.cycle=10; o.warm=1; o.reuse=reuse; o.primary_nodes=1; o.available_alt_nodes=8;
    o.branches=2; o.max_nodes=8; o.min_gain_percent=5;
    o.rank2_probability=.6; o.draftstep_seconds=.1;
    return o;
}
static void seed_base(struct janas_s2_controller *p,unsigned reuse)
{
    struct janas_s2_observation o=observation(0,reuse);
    for(unsigned i=0;i<3;i++) janas_s2_observe(p,&o);
}
static void seed_branch(struct janas_s2_controller *p,unsigned reuse,double eager,
                         double fresh_cost,double cached_cost,unsigned cached_tokens)
{
    struct janas_s2_observation b={0};
    b.cycle=10;b.warm=1;b.reuse=reuse;b.alt_nodes=2;b.eager_seconds=eager;
    struct janas_s2_observation f=observation(1,reuse),c=observation(2,reuse);
    f.full_seconds=fresh_cost;c.full_seconds=cached_cost;
    f.verify_seconds=fresh_cost*.8;c.verify_seconds=cached_cost*.8;
    f.expert_cpu_seconds=fresh_cost*.5;c.expert_cpu_seconds=cached_cost*.5;
    c.delivered=cached_tokens;
    for(unsigned i=0;i<3;i++) {janas_s2_observe(p,&b);janas_s2_observe(p,&f);janas_s2_observe(p,&c);}
}
int main(void)
{
    struct janas_s2_controller p;
    struct janas_s2_offer o=offer(1);
    janas_s2_init(&p,7);p.last_probe=10;
    CHECK(janas_s2_choose(&p,&o).alt_nodes==0); /* Missing evidence preserves MTP. */
    seed_base(&p,1);
    CHECK(janas_s2_choose(&p,&o).alt_nodes==0); /* Head confidence is not benefit evidence. */
    seed_branch(&p,1,.1,1,.4,3);
    struct janas_s2_decision d=janas_s2_choose(&p,&o);
    CHECK(d.alt_nodes==2 && d.reason==JANAS_S2_BENEFICIAL);
    CHECK(d.evidence>=3 && d.matched_comparisons>=3 && d.resident_cpu_seconds>0);
    CHECK(d.predicted_utility>d.baseline_utility*1.05);
    o.reuse=0; CHECK(janas_s2_choose(&p,&o).alt_nodes==0); /* Do not cross occupancy classes. */
    seed_base(&p,0);seed_branch(&p,0,.8,.4,1,1);
    CHECK(janas_s2_choose(&p,&o).alt_nodes==0); /* Many expensive singleton groups. */
    o.reuse=1;o.cycle=80;p.last_probe=80;
    CHECK(janas_s2_choose(&p,&o).alt_nodes==0); /* Stale class is not promoted. */
    o=offer(1);o.branches=1;
    CHECK(janas_s2_choose(&p,&o).alt_nodes==0);
    o.branches=2;o.max_nodes=2;
    CHECK(janas_s2_choose(&p,&o).reason==JANAS_S2_NO_CAPACITY);
    o=offer(1);o.rank2_probability=0;
    CHECK(janas_s2_choose(&p,&o).reason==JANAS_S2_WEAK_CONFIDENCE);
    /* Cached scoring alone cannot repay an eager build paid on all roots. */
    janas_s2_init(&p,2);p.last_probe=10;seed_base(&p,1);seed_branch(&p,1,.7,1,.5,2);
    o=offer(1);CHECK(janas_s2_choose(&p,&o).alt_nodes==0);
    /* Unknown resident CPU cost is refused, even when SSD bytes would be zero. */
    janas_s2_init(&p,3);p.last_probe=10;
    struct janas_s2_observation bad=observation(0,1);bad.expert_cpu_seconds=0;
    for(unsigned i=0;i<3;i++)janas_s2_observe(&p,&bad);
    CHECK(p.width[1][0][1][2].count==0);
    CHECK(janas_s2_choose(&p,&o).alt_nodes==0);
    /* Eager/ cached costs cannot train the ordinary MTP full-cycle table. */
    seed_base(&p,1);double baseline=p.width[1][0][1][2].seconds;
    bad=observation(2,1);bad.full_seconds=10;janas_s2_observe(&p,&bad);
    bad=observation(0,1);bad.alt_nodes=2;bad.eager_seconds=3;bad.full_seconds=8;
    janas_s2_observe(&p,&bad);
    CHECK(p.width[1][0][1][2].seconds==baseline);
    /* Geometry and nonfinite guards reject malformed inputs without mutation. */
    unsigned count=p.width[1][0][1][2].count;
    bad=observation(0,1);bad.full_seconds=NAN;janas_s2_observe(&p,&bad);
    CHECK(p.width[1][0][1][2].count==count);
    o=offer(1);o.rank2_probability=NAN;CHECK(janas_s2_choose(&p,&o).alt_nodes==0);
    CHECK(janas_s2_reuse_class(100,100,8)==0);
    CHECK(janas_s2_reuse_class(100,400,8)==1);
    CHECK(janas_s2_reuse_class(100,800,8)==2);
    CHECK(janas_s2_reuse_class(0,0,8)==3);
    /* Independent probes are sparse and never exceed two initial nodes. */
    janas_s2_init(&p,42);seed_base(&p,1);o=offer(1);
    unsigned probes=0;uint64_t last=0;
    for(o.cycle=11;o.cycle<4096;o.cycle++){
        struct janas_s2_observation b=observation(0,1);b.cycle=o.cycle;
        janas_s2_observe(&p,&b);d=janas_s2_choose(&p,&o);
        if(d.probe){CHECK(d.alt_nodes==2);CHECK(!last || o.cycle-last>=64);last=o.cycle;probes++;}
        else CHECK(d.alt_nodes==0);
    }
    CHECK(probes>0 && probes<=64);
    /* Same private seed and observations reproduce probes; target RNG unused. */
    struct janas_s2_controller q; janas_s2_init(&p,55);janas_s2_init(&q,55);
    seed_base(&p,1);seed_base(&q,1);o=offer(1);
    for(o.cycle=11;o.cycle<200;o.cycle++)CHECK(janas_s2_choose(&p,&o).probe==janas_s2_choose(&q,&o).probe);
    printf("test_s2_controller: %s\n",failures?"FAILED":"ok");return failures?1:0;
}
