from collections import defaultdict, namedtuple
import random
from pprint import pprint

Input = namedtuple('Input', ['state', 'read'])
Result = namedtuple('Result', ['state', 'write', 'direction'])
directions = [-1, 0, +1]


def to_tape(data, blank):
    tape = [blank, blank, *list(data), blank, blank]
    return tape


class Individual:
    def __init__(self, rules=None, num_inputs=None, num_outputs=None):
        if rules is not None:
            self.rules = rules
        else:
            assert num_inputs and num_outputs
            self.rules = random.sample(range(num_outputs), num_inputs)

    def __sub__(self, other):
        diff_vector = []
        for i in range(self.rules):
            diff_vector.append(self.rules[i] - other.rules[i])
        return diff_vector

    def add_noise(self, diffs):
        return Individual(rules=[original + diff
                                 for original, diff
                                 in zip(self.rules, diffs)])


class TransitionTable:
    def __init__(self):
        self._table = {}

    def add_rule(self, input_rule, result):
        self._table[input_rule] = result

    def evaluate(self, input_rule):
        try:
            return self._table[input_rule]
        except:
            raise ValueError('State {} reading {} is not defined'.format(
                input_rule.state, input_rule.read
            ))

    def __str__(self):
        return '{!r}'.format(self._table)


class Turing:
    def __init__(self, states, alphabet, symbols, transition,
                 initial_state, blank, final_states):
        self.states = self.states
        self.alphabet = alphabet
        self.symbols = symbols
        self.transition = transition
        self.initial_state = initial_state
        self.blank = blank
        self.final_states = final_states
        # Classical Optimization Encoding
        self.input_list = [Input(state, symbol)
                           for state in states
                           for symbol in self.alphabet]
        self.output_list = [Result(state, symbol, direction)
                            for state in self.states
                            for symbol in self.symbols
                            for direction in (0, 1, 2)]
        # Differential Evolution
        self.dimension = (len(self.states) - 1) * len(self.symbols)

    def evaluate(self, input_str, output_str):
        state = self.initial_state
        head = 1
        tape = to_tape(input_str, self.blank)
        while state not in self.final_states:
            input_symbol = tape[head]
            result = self.transition.evaluate(Input(state, input_symbol))
            state = result.state
            tape[head] = result.write
            head += directions[result.direction]
        tape_str = ''.join(tape)
        cost_value = 0
        value = 10
        for predicted, real in zip(tape_str, output_str):
            if predicted == real:
                cost_value -= value
            else:
                cost_value += value

    def from_individual(self, individual):
        self.transition = TransitionTable()
        for input_idx, output_idx in enumerate(individual.rules):
            self.transition.add_rule(self.input_list[input_idx],
                                     self.output_list[output_idx])

def evolve_generation(population, F, CR):
    assert population >= 4
    for individual in population:
        surrogates = []
        for i in range(3):
            surrogate = random.choice(population)
            # If we get a dupe, pick again
            while (surrogate == individual or
                   surrogate in surrogates):
                surrogate = random.choice(population)
            surrogates.append(surrogate)
        diff_vector = surrogates[0] - surrogates[1]
        weighted_vector = [diff *  F for diff in diff_vector]
        noise_individual = surrogate[2].add_noise(weighted_vector)
        test_vector = Individual

def classical_opt(self):
    pass

if __name__ == '__main__':
    # Define Turing Machine
    states = []  # Not needed
    alphabet = ['0', '1']
    symbols = ['0', '1', 'X']
    blank = '#'
    transition = TransitionTable()
    transition.add_rule('q1', '1', 'q1', '0', 1)
    transition.add_rule('q1', '0', 'q1', '1', 1)
    transition.add_rule('q1', blank, 'q2', '#', 0)
    turing = Turing(states, alphabet, symbols, transition, 
                    'q1', blank, ['q2'])
    answer = turing.evaluate('100100')
    print(answer)
