import json

import pytest

from experiments.harness_optimization.genome import MUTATION_OPERATORS, HarnessGenome, mutate_genome, seed_genome


def test_canonical_hash_and_deterministic_mutations():
    seed = seed_genome()
    assert seed.candidate_hash == seed_genome().candidate_hash
    for index, operator in enumerate(MUTATION_OPERATORS):
        a = mutate_genome(seed, operator, index)
        b = mutate_genome(seed, operator, index)
        assert a == b
        assert a.candidate_hash == b.candidate_hash != seed.candidate_hash
        a.validate()


def test_rejects_mutable_tool_policy():
    value = seed_genome().to_dict()
    value["tool_policy"]["allow_mutation"] = True
    with pytest.raises(ValueError):
        HarnessGenome(**value).validate()


def test_round_trip():
    genome = mutate_genome(seed_genome(), "pattern", 7)
    restored = HarnessGenome(**json.loads(json.dumps(genome.to_dict())))
    assert restored.candidate_hash == genome.candidate_hash

