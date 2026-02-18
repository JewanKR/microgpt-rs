//! ``` microgpt.py
//! The most atomic way to train and run inference for a GPT in pure, dependency-free Python.
//! This file is the complete algorithm.
//! Everything else is just efficiency.
//!
//! @karpathy
//! ```

use std::{cell::Cell, collections::{HashMap, HashSet}, fs, io::Write, iter::{Sum, once}, ops::{Add, Div, Mul, Neg, Sub}, path, rc::Rc};
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
use rand_distr::{Distribution, Normal, weighted::WeightedIndex};

// [porter's note] Rust requires explicit trait implementations for bidirectional operations.
// I used a macro to mirror Python's flexible arithmetic without sacrificing performance.
macro_rules! impl_value {
    ($t:ident, $m:ident, $op:tt) => {
        impl $t<f64> for Value {        // [porter's note] Value + f64
            type Output = Value;
            fn $m(self, rhs: f64) -> Self::Output { self $op Self::from(rhs) }
        }
        impl $t<Value> for f64 {        // [porter's note] f64 + Value
            type Output = Value;
            fn $m(self, rhs: Value) -> Self::Output { Value::from(self) $op rhs }
        }
    };
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut random = StdRng::seed_from_u64(42);

    // Let there be a Dataset `docs`: Vec<String> of documents (e.g. a list of names)
    if !path::Path::new("./input.txt").exists() {
        let names_url: &str = "https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt";
        let response = reqwest::blocking::get(names_url)?.text()?;
        fs::File::create("./input.txt")?.write_all(response.as_bytes())?;
    };
    let mut docs: Vec<String> = std::fs::read_to_string("./input.txt")? // Vec<String> of documents
        .lines().map(|l| l.trim()).filter(|l| !l.is_empty()).map(|s| s.to_string()).collect();
    docs.shuffle(&mut random);
    println!("num docs: {}", docs.len());

    // Let there be a Tokenizer to translate strings to sequences of integers ("tokens") and back
    let mut uchars: Vec<char> = docs.iter().flat_map(|c| c.chars()).collect();
    uchars.sort_unstable(); uchars.dedup(); // unique characters in the dataset become token ids 0..n-1
    let bos: usize = uchars.len(); // token id for a special Beginning of Sequence (BOS) token
    let vocab_size: usize = uchars.len() + 1; // total number of unique tokens, +1 is for BOS
    let uchars_map: HashMap<char, usize> = uchars.iter().cloned().enumerate().map(|(i, c)| (c, i)).collect();
    println!("vocab size: {}", vocab_size);

    // Let there be Autograd to recursively apply the chain rule through a computation graph
    #[derive(Clone)]
    struct Value(Rc<ValueSlots>); // Rust optimization for memory usage
    struct ValueSlots { data: Cell<f64>, grad: Cell<f64>, children: Vec<Value>, local_grads: Vec<f64> }
    impl Value {
        fn init(data: f64, children: Vec<Value>, local_grads: Vec<f64>) -> Self {
            Self(Rc::new(ValueSlots {
                data: Cell::new(data),      // scalar value of this node calculated during forward pass
                grad: Cell::new(0.0),       // derivative of the loss w.r.t. this node, calculated in backward pass
                children: children,         // children of this node in the computation graph
                local_grads: local_grads,   // local derivative of this node w.r.t. its children
            } ))
        }
    }
    impl From<f64> for Value {
        fn from(value: f64) -> Self { Self::init(value, Vec::new(), Vec::new()) }
    }
    impl Add for Value {
        type Output = Self;
        fn add(self, rhs: Self) -> Self::Output {
            Self::Output::init(self.0.data.get() + rhs.0.data.get(), vec![self.clone(), rhs.clone()], vec![1.0, 1.0])
        }
    }
    impl_value!(Add, add, +);   // [porter's note] add bidirectional operations
    impl Mul for Value {
        type Output = Self;
        fn mul(self, rhs: Self) -> Self::Output {
            let (self_data, rhs_data) = (self.0.data.get(), rhs.0.data.get());
            Self::Output::init(self_data * rhs_data, vec![self.clone(), rhs.clone()], vec![rhs_data, self_data])
        }
    }
    impl_value!(Mul, mul, *);   // [porter's note] add bidirectional operations
    impl Value {
        pub fn powf(self, rhs: f64) -> Self {
            Self::init(self.0.data.get().powf(rhs), vec![self.clone()], vec![rhs * self.0.data.get().powf(rhs - 1.0)])
        }
        pub fn ln(self) -> Self {
            Self::init(self.0.data.get().ln(), vec![self.clone()], vec![1.0 / self.0.data.get()])
        }
        pub fn exp(self) -> Self {
            Self::init(self.0.data.get().exp(), vec![self.clone()], vec![self.0.data.get().exp()])
        }
        pub fn relu(self) -> Self {
            Self::init(self.0.data.get().max(0.0), vec![self.clone()], vec![if self.0.data.get() > 0.0 { 1.0 } else { 0.0 }])
        }
    }
    impl Neg for Value { type Output = Self; fn neg(self) -> Self::Output {self * -1.0} }
    impl Sub for Value { type Output = Self; fn sub(self, rhs: Self) -> Self::Output { self + (-rhs) } }
    impl_value!(Sub, sub, -);   // [porter's note] add bidirectional operations
    impl Div for Value { type Output = Self; fn div(self, rhs: Self) -> Self::Output { self * rhs.powf(-1.0) } }
    impl_value!(Div, div, /);   // [porter's note] add bidirectional operations
    impl Sum for Value {        // [porter's note] Provides sum() for Value
        fn sum<I: Iterator<Item = Self>>(iter: I) -> Self { iter.fold(Value::from(0.0), |acc, val| acc + val) }
    }

    impl Value {
        fn backward(&self) {
            let mut topo = Vec::<Value>::new();
            let mut visited = HashSet::<*const ValueSlots>::new();
            fn build_topo(v: &Value, visited: &mut HashSet<*const ValueSlots>, topo: &mut Vec<Value>) {
                let ptr = Rc::as_ptr(&v.0);
                if !visited.contains(&ptr) {
                    visited.insert(ptr);
                    for child in v.0.children.clone() {
                        build_topo(&child, visited, topo);
                    }
                    topo.push(v.clone())
                }
            }
            build_topo(self, &mut visited, &mut topo);
            self.0.grad.set(1.0);
            for v in topo.into_iter().rev() {
                for (child, local_grad) in v.0.children.iter().zip(v.0.local_grads.iter()) {
                    child.0.grad.set(child.0.grad.get() + local_grad * v.0.grad.get());
                }
            }
        }
    }

    // Initialize the parameters, to store the knowledge of the model.
    const N_LAYER: usize = 1;     // depth of the transformer neural network (number of layers)
    const N_EMBD: usize = 16;     // width of the network (embedding dimension)
    const BLOCK_SIZE: usize = 16; // maximum context length of the attention window (note: the longest name is 15 characters)
    const N_HEAD: usize = 4;      // number of attention heads
    const HEAD_DIM: usize = N_EMBD / N_HEAD; // derived dimension of each head
    type Matrix = Vec<Vec<Value>>;
    fn matrix(nout: usize, nin: usize, std: f64, random: &mut StdRng) -> Matrix {
        let normal = Normal::new(0.0, std).unwrap();
        (0..nout).map(|_| { (0..nin).map(|_| Value::from(normal.sample(random))).collect() }).collect()
    }
    struct StateDict { wte: Matrix, wpe: Matrix, lm_head: Matrix, layers: Vec<SDLayer> }
    struct SDLayer { attn_wq: Matrix, attn_wk: Matrix, attn_wv: Matrix, attn_wo: Matrix, mlp_fc1: Matrix, mlp_fc2: Matrix }
    let state_dict = StateDict {
        wte: matrix(vocab_size, N_EMBD, 0.08, &mut random),
        wpe: matrix(BLOCK_SIZE, N_EMBD, 0.08, &mut random),
        lm_head: matrix(vocab_size, N_EMBD, 0.08, &mut random),
        layers: (0..N_LAYER).map(|_| SDLayer {
            attn_wq: matrix(N_EMBD, N_EMBD, 0.08, &mut random),
            attn_wk: matrix(N_EMBD, N_EMBD, 0.08, &mut random),
            attn_wv: matrix(N_EMBD, N_EMBD, 0.08, &mut random),
            attn_wo: matrix(N_EMBD, N_EMBD, 0.08, &mut random),
            mlp_fc1: matrix(4 * N_EMBD, N_EMBD, 0.08, &mut random),
            mlp_fc2: matrix(N_EMBD, 4 * N_EMBD, 0.08, &mut random),
        } ).collect()
    };
    impl StateDict {
        fn flatten_ref(&self) -> Vec<Value> {
            let mut output = Vec::new();
            let mut flatten = |mat: &Matrix| { output.extend(mat.iter().flatten().cloned()); };
            flatten(&self.wte); flatten(&self.wpe); flatten(&self.lm_head);
            for li in &self.layers {
                flatten(&li.attn_wq);
                flatten(&li.attn_wk);
                flatten(&li.attn_wv);
                flatten(&li.attn_wo);
                flatten(&li.mlp_fc1);
                flatten(&li.mlp_fc2);
            }
            output
        }
    }
    let params: Vec<Value> = state_dict.flatten_ref(); // flatten params into a single Vec<Value>
    println!("num params: {}", params.len());

    // Define the model architecture: a function mapping tokens and parameters to logits over what comes next
    // Follow GPT-2, blessed among the GPTs, with minor differences: layernorm -> rmsnorm, no biases, GeLU -> ReLU
    fn linear(x: &[Value], w: &[Vec<Value>]) -> Vec<Value>{
        w.iter().map(|wo| wo.iter().cloned().zip(x.iter().cloned()).map(|(wi, xi)| wi * xi).sum()).collect()
    }

    fn softmax(logits: &[Value]) -> Vec<Value> {
        let max_val = logits.iter().map(|val| val.0.data.get()).max_by(|a, b| a.total_cmp(b)).unwrap_or(0.0);
        let exps: Vec<Value> = logits.iter().cloned().map(|val| (val - max_val).exp()).collect();
        let total:Value = exps.iter().cloned().sum();
        exps.into_iter().map(|e| e / total.clone()).collect()
    }

    fn rmsnorm(x: &[Value]) -> Vec<Value> {
        let ms = x.iter().cloned().map(|xi| xi.clone() * xi).sum::<Value>() / Value::from(x.len() as f64);
        let scale = (ms + 1e-5).powf(-0.5);
        x.iter().cloned().map(|xi| xi * scale.clone()).collect()
    }

    impl StateDict {
        fn gpt(&self, token_id: usize, pos_id: usize, keys: &mut [Matrix], values: &mut [Matrix]) -> Vec<Value> {
            let tok_emb: Vec<Value> = self.wte[token_id].clone(); // token embedding
            let pos_emb: Vec<Value> = self.wpe[pos_id].clone(); // position embedding
            let mut x: Vec<Value> = tok_emb.into_iter().zip(pos_emb.into_iter()).map(|(t, p)| t + p).collect(); // joint token and position embedding
            x = rmsnorm(&x); // note: not redundant due to backward pass via the residual connection

            for (layer, (keys_li, valuse_li)) in self.layers.iter().zip(keys.iter_mut().zip(values.iter_mut())) {
                // 1) Multi-head Attention block
                let x_residual: Vec<Value> = x.clone();
                x = rmsnorm(&x);
                let q: Vec<Value> = linear(&x, &layer.attn_wq);
                let k: Vec<Value> = linear(&x, &layer.attn_wk);
                let v: Vec<Value> = linear(&x, &layer.attn_wv);
                keys_li.push(k);
                valuse_li.push(v);
                let mut x_attn = Vec::new();
                for h in 0..N_HEAD{
                    let hs = h * HEAD_DIM;
                    let q_h = &q[hs..hs+HEAD_DIM];
                    let k_h: Matrix = keys_li.iter().map(|ki| ki[hs..hs+HEAD_DIM].to_vec()).collect();
                    let v_h: Matrix = valuse_li.iter().map(|vi| vi[hs..hs+HEAD_DIM].to_vec()).collect();
                    let attn_logits: Vec<Value> = k_h.into_iter().map(|k_ht| k_ht.into_iter().zip(q_h.iter())
                        .map(|(k_htj, q_hj)| q_hj.clone() * k_htj).sum::<Value>() / (HEAD_DIM as f64).powf(0.5))
                        .collect();
                    let attn_weights: Vec<Value> = softmax(&attn_logits);
                    let head_out: Vec<Value> = v_h.into_iter().map(|v_ht| v_ht.into_iter().zip(attn_weights.iter().cloned())
                        .map(|(v_htj, aw)| aw * v_htj).sum::<Value>())
                        .collect();
                    x_attn.extend(head_out);
                }
                x = linear(&x_attn, &layer.attn_wo);
                x = x.into_iter().zip(x_residual.into_iter()).map(|(a, b)| a + b).collect();
                // 2) MLP block
                let x_residual: Vec<Value> = x.clone();
                x = rmsnorm(&x);
                x = linear(&x, &layer.mlp_fc1);
                x = x.into_iter().map(|xi| xi.relu()).collect();
                x = linear(&x, &layer.mlp_fc2);
                x = x.into_iter().zip(x_residual.into_iter()).map(|(a, b)| a + b).collect();
            }
            linear(&x, &self.lm_head)
        }
    }

    // Let there be Adam, the blessed optimizer and its buffers
    let (learning_rate, beta1, beta2, eps_adam) = (0.01, 0.85, 0.99, 1e-8);
    let mut m = vec![0.0; params.len()]; // first moment buffer
    let mut v = vec![0.0; params.len()]; // second moment buffer

    // Repeat in sequence
    let num_steps = 1000; // number of training steps
    for step in 0..num_steps {
        // Take single document, tokenize it, surround it with BOS special token on both sides
        let doc = &docs[step % docs.len()];
        let tokens: Vec<usize> = once(bos).chain(doc.chars().filter_map(|c| uchars_map.get(&c).copied())).chain(once(bos)).collect();
        let n: usize = BLOCK_SIZE.min(tokens.len() - 1);

        // Forward the token sequence through the model, building up the computation graph all the way to the loss
        let (mut keys, mut values): (Vec<Matrix>, Vec<Matrix>) = (Vec::with_capacity(N_LAYER), Vec::with_capacity(N_LAYER));
        let mut losses: Vec<Value> = Vec::new();
        for pos_id in 0..n {
            let (token_id, target_id) = (tokens[pos_id], tokens[pos_id + 1]);
            let logits = state_dict.gpt(token_id, pos_id, &mut keys, &mut values);
            let probs = softmax(&logits);
            let loss_t = -probs[target_id].clone().ln();
            losses.push(loss_t);
        }
        let loss: Value = (1.0 / n as f64) * (losses.into_iter().sum::<Value>()); // final average loss over the document sequence. May yours be low.

        // Backward the loss, calculating the gradients with respect to all model parameters
        loss.backward();

        // Adam optimizer update: update the model parameters based on the corresponding gradients
        let lr_t = learning_rate * (1.0 - step as f64 / num_steps as f64); // linear learning rate decay
        for (p,(mi, vi)) in params.iter().zip(m.iter_mut().zip(v.iter_mut())) {
            *mi = beta1 * *mi + (1.0 - beta1) * p.0.grad.get();
            *vi = beta2 * *vi + (1.0 - beta2) * p.0.grad.get().powf(2.0);
            let m_hat = *mi / (1.0 - beta1.powf(step as f64 + 1.0));
            let v_hat = *vi / (1.0 - beta2.powf(step as f64 + 1.0));
            p.0.data.set(p.0.data.get() - (lr_t * m_hat / (v_hat.powf(0.5) + eps_adam)));
            p.0.grad.set(0.0);
        }
        print!("\rstep {:4} / {:4} | loss {:.4}", step+1, num_steps, loss.0.data.get());
    }

    // Inference: may the model babble back to us
    let temperature = 0.5; // in (0, 1], control the "creativity" of generated text, low to high
    println!("\n--- inference (new, hallucinated names) ---");
    for sample_idx in 0..20 {
        let (mut keys, mut values): (Vec<Matrix>, Vec<Matrix>) = (Vec::with_capacity(N_LAYER), Vec::with_capacity(N_LAYER));
        let mut token_id = bos;
        let mut sample = Vec::new();
        for pos_id in 0..BLOCK_SIZE {
            let logits: Vec<Value> = state_dict.gpt(token_id, pos_id, &mut keys, &mut values);
            let probs: Vec<Value> = softmax(&logits.into_iter().map(|l| l / temperature).collect::<Vec<Value>>());
            token_id = WeightedIndex::new(&probs.into_iter().map(|p| p.0.data.get()).collect::<Vec<f64>>())?.sample(&mut random);
            if token_id == bos {break;}
            sample.push(uchars.get(token_id).copied().unwrap_or('?'));
        }
        println!("sample {:2}: {}", sample_idx+1, String::from_iter(sample));
    }
    Ok(())
}
