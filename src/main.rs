// I applied techniques I learned from Reinforcement Learning: An Introduction by Richard S. Sutton and Andrew G. Barto
// 	https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf

use rand::distributions::Open01;
use rand::Rng;
use std::fmt::{self, Debug, Formatter};
use std::ops::Index;
use std::ops::IndexMut;

struct Commutative2DArray<T> {
	vec: Vec<Vec<T>>
}

impl<T> Commutative2DArray<T> {
	fn new<F>(size: usize, mut new_element: F) -> Commutative2DArray<T> where F: FnMut(usize, usize) -> T {
		Commutative2DArray {
			vec: (0..size).map(|i| (0..=i).map(|j| new_element(i, j)).collect()).collect()
		}
	}
}

impl<T: Debug> Debug for Commutative2DArray<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.vec)
    }
}

impl<T> Index<(usize, usize)> for Commutative2DArray<T> {
	type Output = T;

	fn index(&self, (i, j): (usize, usize)) -> &T {
		&self.vec[std::cmp::max(i, j)][std::cmp::min(i, j)]
	}
}

// For convenience
impl<T> Index<(u8, u8)> for Commutative2DArray<T> {
	type Output = T;

	fn index(&self, (i, j): (u8, u8)) -> &T {
		&self.vec[std::cmp::max(i, j) as usize][std::cmp::min(i, j) as usize]
	}
}

impl<T> IndexMut<(usize, usize)> for Commutative2DArray<T> {
	fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut T {
		&mut self.vec[std::cmp::max(i, j)][std::cmp::min(i, j)]
	}
}

// For convenience
impl<T> IndexMut<(u8, u8)> for Commutative2DArray<T> {
	fn index_mut(&mut self, (i, j): (u8, u8)) -> &mut T {
		&mut self.vec[std::cmp::max(i, j) as usize][std::cmp::min(i, j) as usize]
	}
}

struct Model {
	// Empirical probability of winning per each state
	probabilities: Commutative2DArray<Commutative2DArray<f32>>,

	// Sample size per each state
	sample_sizes: Commutative2DArray<Commutative2DArray<u32>>
}

impl Model {
	fn new() -> Model {
		let mut rng = rand::thread_rng();

		Model {
			// Initialize each probability to a random one
			probabilities: Commutative2DArray::new(5,
				|home0, home1| Commutative2DArray::new(5, |away0, away1| match (home0, home1, away0, away1) {
					(0, 0, 0, 0) => f32::NAN,
					(0, 0, _, _) => 0.0,
					(_, _, 0, 0) => 1.0,
					_ => rng.sample(Open01)
				})
			),

			// Initialize each sample size to one so as to avoid a probability of 0 or 1 (which would inhibit exploration)
			sample_sizes: Commutative2DArray::new(5, |_, _| Commutative2DArray::new(5, |_, _| 1))
		}
	}

	// Predicts the most optimal move from a given state
	//
	// A state is a 2x2 tuple corresponding to ((home1, home2), (away1, away2)), where:
	// 	home1 and home 2 are the model's hands; away1 and away2 are the opponent's hands
	// 	home1 >= home2 and away1 >= away2
	//
	// deterministic should only be used when training (it enables the model to explore non-optimal states)
	fn predict(&self, (home, away): ((u8, u8), (u8, u8)), deterministic: bool) -> Option<((u8, u8), (u8, u8))> {
		let mut possible_states = Vec::new();

		// Consider every combination of transferring one's chopsticks
		if away.0 > 0 {
			if home.0 > 0 {
				possible_states.push((home, (away.0 + home.0, away.1)));
			}

			if home.1 > 0 {
				possible_states.push((home, (away.0 + home.1, away.1)));
			}
		}

		if away.1 > 0 {
			if home.0 > 0 {
				possible_states.push((home, (away.0, away.1 + home.0)));
			}

			if home.1 > 0 {
				possible_states.push((home, (away.0, away.1 + home.1)));
			}
		}

		// Set hands greater than four to zero
		for (_, (i, j)) in &mut possible_states {
			if *i > 4 {
				*i = 0;
			}

			if *j > 4 {
				*j = 0;
			}
		}

		// Consider every combination of reallocating one's chopsticks
		for i in 0.max(home.0 as i8 + home.1 as i8 - 4) as u8..=(home.0 + home.1) / 2 {
			// Disallow swapping hands (it's effectiveless and never strategic)
			if i != home.0 && i != home.1 {
				possible_states.push(((i, home.0 + home.1 - i), away))
			}
		}

		if possible_states.len() == 0 {
			return None;
		}

		// Sort the possible new states by descending probability of winning
		possible_states.sort_unstable_by(|(home0, away0), (home1, away1)|
			self.probabilities[*home1][*away1].partial_cmp(&self.probabilities[*home0][*away0]).unwrap());

		let mut rng = rand::thread_rng();

		Some(if deterministic {
			possible_states[0]
		} else {
			// I've concevied of two methods of suboptimal state exploration
			*possible_states[0..possible_states.len()]
				.iter()
				// .find(|(new_home, new_away)| rng.gen::<f32>() < self.probabilities[*new_home][*new_away])
				.find(|_| rng.gen::<f32>() < 0.6)
				.unwrap_or_else(|| possible_states.last().unwrap())
		})
	}

	// predict from the perspective of the opponent
	fn predict_reversed(&self, (away, home): ((u8, u8), (u8, u8)), deterministic: bool) -> Option<((u8, u8), (u8, u8))> {
		let (new_home, new_away) = self.predict((home, away), deterministic)?;
		Some((new_away, new_home))
	}

	// Train the model using every state in which it was before and after it won/lost
	fn fit(&mut self, states: &[((u8, u8), (u8, u8))], won: bool) {
		for (home_state, away_state) in states {
			let p = &mut self.probabilities[*home_state][*away_state];
			let ss = &mut self.sample_sizes[*home_state][*away_state];

			// https://math.stackexchange.com/questions/106700/incremental-averageing
			*p = (*p * *ss as f32 + won as u8 as f32) / (*ss as f32 + 1.0);
			*ss += 1;
		}
	}

	// fit from the perspective of the opponent
	fn fit_reversed(&mut self, states: &[((u8, u8), (u8, u8))], won: bool) {
		self.fit(&states.iter().map(|state| (state.1, state.0)).collect::<Vec<_>>(), won);
	}
}

fn main() {
	let mut rng = rand::thread_rng();

	// Train two models against each other via 5,000 games
	let (mut model0, mut model1) = (Model::new(), Model::new());

	for _ in 0..5000 {
		// Both models begin with one finger on each hand
		// 	The first element of every state corresponds to model0; the second to model1
		let mut states = vec![((1, 1), (1, 1))];

		// false or true respectively encode whether it is model0 or model1's turn
		//	Initialized randomly to prevent bias
		let mut turn: bool = rng.gen();

		loop {
			states.push(if turn {
				model1.predict_reversed(*states.last().unwrap(), false)
			} else {
				model0.predict(*states.last().unwrap(), false)
			}.unwrap());

			if states.last().unwrap().0 == (0, 0) {
				model0.fit(&states, false);
				model1.fit_reversed(&states, true);

				break;
			}

			if states.last().unwrap().1 == (0, 0) {
				model0.fit(&states, true);
				model1.fit_reversed(&states, false);

				break;
			}

			turn = !turn;
		}
	}

	println!("Training complete.");

	loop {
		let mut line = String::new();
		std::io::stdin().read_line(&mut line).unwrap();

		let state: Vec<u8> = line
			.trim_end()
			.split(" ")
			.map(|s| s.parse().unwrap())
			.collect();

		println!("{:?}", model0.predict_reversed(((state[0], state[1]), (state[2], state[3])), true));
	}
}
