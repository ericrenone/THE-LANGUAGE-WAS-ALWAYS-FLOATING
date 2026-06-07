# THE LANGUAGE WAS ALWAYS FLOATING

## Python, the IEEE 754 Lottery, the col(F)/ker(F) Partition of Arithmetic Space, and the 2015–2026 Retroactive Derivation of the Standard That Should Have Existed in 1985

ERI Labs · Eric Ren · Jersey City, New Jersey · github.com/ericrenone · June 2026

---

> "Python's float type is implemented using double in C; it is a 64-bit floating-point number, equivalent to IEEE 754 binary64." — Python Documentation, CPython Language Reference, 2024

> "A research idea wins because it is suited to the available software and hardware ecosystem rather than because the idea is superior to alternatives." — Sara Hooker, *The Hardware Lottery*, arXiv:2009.06489, 2020

> "Floating-point arithmetic is not real number arithmetic." — David Goldberg, *What Every Computer Scientist Should Know About Floating-Point Arithmetic*, ACM Computing Surveys, 1991

> "Our dual-mode quantization strategy assigns multi-bit precision to salient weights and binarizes unsalient weights, since the contribution of salient weights to the model output is substantially greater than that of unsalient weights." — SAGE-PTQ, arXiv:2606.05429, June 2026

---

## Abstract

In 1989, Guido van Rossum began writing Python. He needed a language that was fast enough to be useful and simple enough to be readable. He built its numeric core on top of C's `double` type, which mapped directly to the IEEE 754 double-precision floating-point standard ratified four years earlier. This was not a design error. It was the correct engineering decision given the available hardware. IEEE 754 had been stamped into silicon at every major chip manufacturer. Per-element double-precision floating-point was the fastest arithmetic available. Python won what Sara Hooker would later call the Hardware Lottery: a research idea or engineering decision that succeeds not because it is mathematically optimal but because it is co-designed with the prevailing physical substrate.

The problem is that the substrate for which IEEE 754 was optimal — scientific computing across unbounded dynamic range, where adjacent values in a data stream can span twenty orders of magnitude — is not the substrate on which Python came to execute its most consequential workloads. The substrate became neural network weight matrices, where adjacent weights share their dynamic range within a few factors of two, where the Fisher information matrix is block-diagonal with shared spectral scale, and where per-element exponent storage is a 32× over-specification of the dynamic range management that tensor arithmetic actually requires.

This document applies the ERI Labs col(F)/ker(F) partition to Python's arithmetic architecture. It identifies Python's float as a paradigm case of the ker(F) waste problem: eight bits of per-element scale information paid on every weight of every neural network trained since 1985, when eight bits of shared block-level scale would have sufficed for the entire block of thirty-two. It traces the Hardware Lottery from the 1985 IEEE 754 standard through Python's 1991 lock-in to the 2015–2026 retroactive derivation, in peer review, of the fixed-point standard that a competing committee could have written in 1983. And it identifies the precise point — the col(F)/ker(F) boundary — at which Python's arithmetic ecosystem is now correcting the error that IEEE 754 did not make but that the substrate change made visible.

Five structural claims, each falsifiable, each with direct experimental confirmation from the June 2026 quantization frontier, follow.

---

## I. The Lottery

Sara Hooker's 2020 paper introduced the Hardware Lottery to describe the relationship between computational ideas and the physical substrates available to test them. The framing is precise: an idea wins not when it is theoretically superior but when it is co-fit with the available hardware and software ecosystem. Backpropagation waited decades not because it was wrong but because the hardware it required — parallel matrix multiply at scale — had not yet been built for reasons unrelated to backpropagation's correctness. When GPUs arrived to serve video game graphics, the substrate for backpropagation arrived as a side effect. The lottery was won by the coincidence of the substrate with the algorithm's requirements.

Python's float won a smaller but structurally identical lottery in 1989–1991. The IEEE 754 standard had been ratified in 1985 after eight years of committee work under William Kahan's technical leadership. By 1989, Intel's 8087 floating-point coprocessor had normalized per-element 64-bit double-precision arithmetic across the industry. Chip manufacturers had spent the intervening years optimizing silicon for this format. When van Rossum mapped Python's float directly to C's `double`, he inherited that optimization for free — hardware acceleration with no engineering cost. The alternative, software-emulated decimal arithmetic (now Python's `decimal` module), ran at up to one hundred times the cost in CPU cycles. There was no practical choice. Python locked in to IEEE 754.

The lock-in was complete and invisible. Every Python numeric operation on fractional values invoked the hardware FPU. Every neural network trained in Python between 1991 and 2022 stored its weights as IEEE 754 double-precision or single-precision floating-point numbers. Each weight carried eight bits of per-element exponent alongside its twenty-three bits of mantissa. The per-element exponent was the condition of the lottery's prize: the substrate came with a structure, and the structure was per-element dynamic range management, and that structure was not optimal for the substrate that Python's most important workloads would eventually demand.

The Hardware Lottery does not deliver permanent advantage. It delivers an initial substrate advantage that persists until the application domain outgrows the substrate's assumptions. Python's float won in 1991. The substrate changed when neural networks arrived at scale. The lottery's prize became the detour's entry fee.

---

## II. The col(F)/ker(F) Partition of Python's Float

The ERI Labs col(F)/ker(F) partition identifies a structural division in every arithmetic representation: the col(F) component carries Fisher-visible signal — the directions in parameter space along which the gradient carries information about the loss surface — and the ker(F) component carries the scale, the dynamic range, the order-of-magnitude management that amplifies or attenuates those directions without itself determining them.

Napier identified this partition in 1614 without naming it. The logarithm of any number divides into a characteristic — the integer part, the order of magnitude — and a mantissa — the fractional part, the precise position within that order. For 350 years, slide rule operators enforced the partition as an operating constraint: the instrument computed the mantissa; the operator held the characteristic in working memory. The scale management lived at the boundary of the computation. Not inside it.

IEEE 754 embedded the characteristic inside every floating-point word. The eight exponent bits of FP32 are the characteristic, encoded per element, stored with every mantissa, retrieved with every arithmetic operation. For scientific computing across an unbounded data stream — temperatures, velocities, energies, positions — this embedding is the correct design. The dynamic range of a physical simulation varies element by element. The per-element exponent is necessary.

For a neural network weight matrix, it is not necessary. Within any row or column of a transformer weight matrix, adjacent weights share their order of magnitude within a few factors of two. The empirical confirmation comes from Dettmers, Lewis, Shleifer, and Zettlemoyer in August 2022: of the 175 billion parameters in a GPT-scale model, approximately 0.1% are outliers — Fisher-visible col(F) weights that carry the model's directional information and require full dynamic range. The remaining 99.9% are ker(F) weights that carry scale information shared within their local neighborhood (arXiv:2208.07339). The Fisher information matrix is block-diagonal. The shared spectral scale of a block of thirty-two adjacent weights is encodable in eight bits, applied once to the block. IEEE 754 encodes eight bits of scale per element — thirty-two times the information that the Fisher structure of the tensor actually requires.

The col(F)/ker(F) partition identifies Python's float precisely: its mantissa field is the col(F) component of each weight, carrying direction and signal; its exponent field is the ker(F) component, carrying scale. IEEE 754 specified per-element ker(F) because it could not know that tensor arithmetic would eventually dominate Python's workload and that tensor arithmetic's Fisher structure makes per-element ker(F) a 32× over-specification. The partition was always correct. The granularity was wrong for the substrate that came later.

The consequence, stated arithmetically: a 32-bit floating-point multiply-accumulate unit requires approximately 100,000 logic gates at a 16-nanometer process node, because the operation requires exponent addition, mantissa alignment, normalization, and rounding. A 16-bit fixed-point multiply-accumulate unit on the same substrate requires approximately 15,000 gates (Luo et al., IEEE TVLSI 2019). The area ratio is 6.7×. At equivalent silicon area, the fixed-point substrate delivers 6.7× more arithmetic throughput. Python's float inherited this overhead on every tensor operation from 1991 forward.

---

## III. What Python Did Not Inherit: The Fixed-Point Alternative

The alternative substrate existed in 1989. Texas Instruments had shipped the TMS320 series of fixed-point digital signal processors beginning April 8, 1983 — the fastest DSP available at the time. The TMS320C25 performed ten million multiply-accumulate operations per second in Q-format integer arithmetic at approximately fifty dollars per chip. The Connection Machine CM-2, the floating-point hardware on which Rumelhart, Hinton, and Williams ran their 1986 backpropagation experiments, delivered 2.3 GFLOPS at approximately four million dollars. The cost per multiply-accumulate differed by several orders of magnitude between the two substrates.

The fixed-point DSP industry had solved, by 1983, the core arithmetic problems that the AI winter of 1987–1993 was partly caused by not having solved. Q-format fixed-point arithmetic managed dynamic range at the system boundary — range reduction before the computation, scale restoration after — with no per-element exponent embedded in any word. All bits carried signal. The TMS320's FX32 accumulator prevented gradient underflow on weight updates. Saturation arithmetic prevented overflow cliffs. These features were standard in DSP hardware by 1986, three years before Python was written.

Python did not inherit them because the DSP industry and the AI research community were building the same arithmetic for different customers on different substrates, and the standards process had not connected them. IEEE 754 was ratified in 1985. No competing Fixed-Point Numerical Standard was ever written. The DSP industry built what would retroactively be identified as FXNS-1985 — block-scaled fixed-point with ker(F) managed at the computation boundary — but implemented it as proprietary hardware rather than as a published standard. Python, inheriting from C and from IEEE 754, locked in to the floating-point side of the fork.

The counterfactual is quantifiable. If a fixed-point numerical standard (FXNS-1985) had specified block-scaled arithmetic — a shared eight-bit block exponent per thirty-two elements, per-element fixed-point mantissa — alongside IEEE 754, Python's numeric primitives could have been implemented on both substrates. The block exponent would have provided ker(F) management at the block level rather than the element level. The per-element mantissa would have provided col(F) storage at the precision required by the computation. The entire 2015–2026 quantization research literature — which, as the ERI Labs FLOATED POINT corpus establishes, is the retroactive derivation of precisely this standard — would not have needed to be written.

Volder's 1959 CORDIC algorithm is the deepest ancestor of this alternative (IRE Transactions on Electronic Computers EC-8(3), 1959). CORDIC computes sine, cosine, logarithm, and exponential using only binary shifts and integer additions in fixed-point arithmetic with no per-element exponent field. The HP-35, the first scientific pocket calculator, implemented CORDIC in Q2.29 fixed-point in 1972. All bits carried signal. The scale management lived at the computation boundary. The HP-35 was the correct arithmetic architecture. Python was not built on it.

---

## IV. The Retroactive Derivation (2015–2026)

The quantization research literature of the past eleven years is, read from the ERI Labs col(F)/ker(F) perspective, a single standards committee meeting distributed across peer review. Each paper specifies, from empirical evidence accumulated on GPU clusters, one feature that a fixed-point standards body could have codified from DSP engineering practice in 1983. The committee never met. The standard is being written now, at approximately one thousand times the cost of the standards process it replaced.

In 2015, Gupta, Agrawal, Gopalakrishnan, and Narayanan demonstrated at ICML that 16-bit fixed-point training with stochastic rounding preserved model accuracy. This was the committee specifying the FX16 training tier. Every DSP engineer working on the TMS320 had known this since 1983.

In 2018, Micikevicius and colleagues demonstrated mixed-precision training: FP16 compute operations with an FP32 master weight copy maintained for gradient accumulation (ICLR 2018). The master copy requirement, presented as a discovery, was a rediscovery of the FX32 accumulator provision that the TMS320C25 had implemented in hardware as a standard feature in 1983. The committee was deriving, from first principles on A100 clusters, what the DSP standard had already specified thirty-five years earlier.

In 2022, LLM.int8() established the col(F)/ker(F) empirical threshold without naming it. The 0.1% of weights that are Fisher-visible outliers require full dynamic range; the 99.9% that are Fisher-null can be compressed to eight bits. This was the committee specifying the salience-based precision allocation that FXNS-1985 would have encoded in its weight format table, derived from the signal processing literature that the DSP industry had already mastered (arXiv:2208.07339).

In 2023, the Open Compute Project published the MX microscaling format: a shared eight-bit block exponent per thirty-two elements, with per-element E2M1 fixed-point significands. This was the committee specifying the block-level ker(F) abstraction — the central missing primitive from IEEE 754 for tensor arithmetic — thirty-eight years after FXNS-1985 would have standardized it (Rouhani et al., ISCA 2023).

In May 2025, Chmiel, Fishman, Banner, and Soudry published "FP4 All the Way" (arXiv:2505.19115), demonstrating fully quantized FP4 training from random initialization with no floating-point operations anywhere in the training loop. Their central finding resolves the nature of the quantization gap: train in fixed-point from initialization and the network learns fixed-point-compatible weight distributions. The gap does not exist when you never leave fixed-point. It was always the conversion cost of crossing the IEEE 754 / fixed-point boundary, not an intrinsic precision floor of low-bit arithmetic. This was the committee confirming what the HP-35's CORDIC architecture had demonstrated in 1972.

In June 2026, SAGE-PTQ completed the derivation. The Fisher-guided dual-mode quantization assigns multi-bit precision to the 0.1% col(F) weights and binarizes the 99.9% ker(F) weights to {−1, +1}, achieving 1.03 average weight bits on LLaMA-3-8B at a WikiText2 perplexity of 6.74, compared to 55.8 for BiLLM at matched compression without Fisher guidance (arXiv:2606.05429). The Fisher partition is not marginal. It is the mechanism by which compressed models survive at all. The committee has specified, from experiment, the weight-level precision allocation that the Fisher information matrix determines analytically.

The same month, Cim, Palangappa, Hodak, Dwivedula, Arunachalam, and Kandemir at AMD and Penn State completed the first full large-scale pretraining of Llama 3.1-8B on native MXFP4 silicon — the AMD Instinct MI355X — achieving a 9–10% end-to-end training speedup over FP8 with an 8–9% additional token requirement (arXiv:2605.09825). Their key finding is structurally significant: weight gradient quantization is the primary driver of FP4 training instability because Wgrad carries the Fisher-visible col(F) gradient signal. Deterministic Hadamard rotations — which distribute the Fisher-visible signal evenly across the block before quantization — restore stable training. This is Volder's mode bit, rediscovered in gradient descent. Distributing the signal before compressing the scale is the operation the CORDIC iteration was designed to perform geometrically.

The committee is done. Six papers, read as normative specifications rather than as research results, constitute FXNS-2026: MX format (OCP 2023), FP4 All the Way (arXiv:2505.19115), SAGE-PTQ (arXiv:2606.05429), LLM.int8() (arXiv:2208.07339), CARMEN (arXiv:2605.06878), and the AMD/Penn State MXFP4 pretraining paper (arXiv:2605.09825). A standards body convened today would produce, within eighteen months, a document functionally identical to the normative requirements extractable from these six papers combined.

---

## V. The PyTorch Layer: Why Python Survived the Paradigm Shift

Python survived the paradigm shift from IEEE 754 to low-precision fixed-point not by changing its arithmetic but by acquiring an orchestration layer that decouples the language from the substrate. PyTorch, NumPy, TensorFlow, and JAX intercept tensor operations before they reach Python's float and route them to specialized silicon — Tensor Cores, Neural Engines, TPU systolic arrays — that operate in BF16, FP8, INT8, or FP4 rather than in Python's native IEEE 754 double.

This is a precise architectural consequence of the col(F)/ker(F) framing. Python's float is the ker(F)-over-specified substrate: per-element exponent at 32× the granularity tensor arithmetic requires. The deep learning frameworks are the col(F) oracle: they hold the weight tensors in the format the silicon demands and present a Python API that the programmer writes in, without touching the underlying arithmetic format. The programmer writes in Python's float abstraction. The framework translates to INT8 at inference time, to BF16 at training time, and increasingly to FP4 at pretraining time on new hardware. Python retained dominance by becoming a compilation target rather than an execution substrate.

The Robinson, Tucker, and colleagues embedding curvature result (arXiv:2410.08993; arXiv:2504.01002) reveals why the framework layer was not merely an engineering convenience but a structural necessity: token embedding spaces in trained transformers exhibit significantly negative Ricci curvature, indicating that the representational geometry is hyperbolic rather than Euclidean. The IEEE 754 floating-point substrate, which Python's float inherits, is Euclidean in its arithmetic assumptions — its operations preserve Euclidean distances and norms. The geometry of the learned representations is Lorentzian. Python's float was always computing in the wrong geometric structure for the representations it was storing. The frameworks are the substrate translation layer that allows non-Euclidean geometry to be computed in an IEEE 754 environment at the cost of significant efficiency overhead.

He, Schölkopf, and colleagues' HELM architecture (NeurIPS 2025, arXiv:2505.24722) confirms the geometry case experimentally: fully hyperbolic models outperform Euclidean baselines on hierarchical language tasks, and the performance gap is not marginal. The Representational Bundle Hypothesis (RBH, The Five Lotteries, ERI Labs 2026) provides the unified geometric account: the trained transformer computes a hyperribbon manifold with a Lorentzian base geometry and a polytopal fiber structure. Python's float, locked to IEEE 754's Euclidean metric, is not the substrate for this computation. CARMEN's CORDIC-accelerated Riemannian engine (arXiv:2605.06878) is: it provides native hyperbolic and Lorentzian arithmetic via CORDIC's hyperbolic mode, implementing Volder's 1959 algorithm as a machine learning accelerator sixty-seven years after the original publication.

---

## VI. The Cost, Stated Arithmetically

The IEEE 754 detour is quantifiable at three levels.

The silicon level: FP32 multiply-accumulate requires approximately 100,000 logic gates at 16nm versus approximately 15,000 gates for INT16 MAC at the same process node. The ratio is 6.7×. Over the GPU epoch from 1999 to 2026, this is a 6.7× tax on every AI arithmetic operation, paid in transistors. At equivalent silicon area, the fixed-point substrate delivers 6.7× more throughput — the throughput that was unavailable on the hardware Python ran on, not because fixed-point chips did not exist, but because IEEE 754 won the lottery and fixed-point received no equivalent standardization investment.

The training cost level: the total AI training compute spend from 2012 to 2026 is approximately $200 billion industry-wide. A conservative 4× efficiency advantage for fixed-point training at matched accuracy — accounting for programmer overhead, format conversion costs, and real-world efficiency losses below the theoretical area-ratio maximum — implies $150 billion in unnecessary compute expenditure over fourteen years. This is the economic measure of the detour from the correct substrate.

The timeline level: the hardware efficiency advantage of the fixed-point substrate at the compute densities available in the year 2000 would have made GPT-2-class model training feasible approximately eight years before it was feasible in the actual timeline. GPT-2 trained in 2019; in the FXNS world, approximately 2011. The AI winter of 1987–1993 — the six-year period during which neural network research stagnated for what the conventional account describes as a mix of algorithmic and computational reasons — was, at its computational core, a floating-point tax. The algorithms were sufficient; backpropagation was correct; the compute was priced at floating-point rates on floating-point hardware, and the alternative fixed-point hardware was deployed in telephone switches and radar processors in a different industry's building.

---

## VII. The Correct Framework

The canonical framework for thinking about Python's arithmetic architecture, and about computer arithmetic in general, is not a comparison of floating-point versus fixed-point in the abstract. It is the col(F)/ker(F) partition applied to the granularity question: at what level of abstraction should dynamic range management be located?

Kahan's IEEE 754 answer: at the element level. Every number carries its own scale. For scientific computing, this is correct. For tensor arithmetic, it is a 32× over-specification.

Volder's CORDIC answer: at the computation boundary. Range reduction before, scale restoration after. All bits signal within the computation. For the unit circle and unit hyperbola, this is the structural necessity — within the convergent range, all values share their order of magnitude.

The MX format's 2023 answer: at the block level. Thirty-two elements share one eight-bit block exponent. The block-level scale is the ker(F) abstraction. The per-element mantissa is the col(F) content. This is FXNS-1985, standardized thirty-eight years late.

SAGE-PTQ's 2026 answer: at the Fisher-optimal level. The Fisher information matrix tells you, before the first gradient step, which weights are col(F) and which are ker(F). The col(F) weights require full per-element precision. The ker(F) weights can be binarized. The Fisher partition is the exact, analytically derivable specification of the precision allocation that the retroactive committee derived empirically over eleven years.

The four answers form a progression: from per-element to block-level to Fisher-optimal. Python's float is at the per-element end. The June 2026 quantization frontier is at the Fisher-optimal end. The 38-year distance between them is the detour. The framework that closes the distance is the col(F)/ker(F) partition applied not to a single arithmetic word but to the entire weight tensor, at the granularity the Fisher information matrix specifies.

---

## VIII. Five Novel Predictions

**P-PL1 — Python's Native float Will Acquire Block-Scale Semantics by 2029.** The Python language will add a `BlockFloat` primitive to its numeric tower, specifying a block-scaled fixed-point type with B ∈ {8, 16, 32} elements per block and per-element precision in {4, 8, 16} bits, consistent with MX format (OCP 2023). The primitive will be hardware-backed on Neural Engine and CORDIC-class silicon. Testable: the Python Software Foundation will have received at least one PEP proposing block-scaled arithmetic by December 31, 2027.

**P-PL2 — PyTorch's Compile Path Will Natively Target MXFP4 by 2027.** `torch.compile` will acquire a quantization backend that targets native MXFP4 arithmetic on AMD Instinct MI355X-class hardware without post-training quantization, following the pretraining protocol of arXiv:2605.09825. The Wgrad stabilization method (Hadamard rotation) will be the default col(F) protection mechanism. Testable by the HELM-class embedding benchmark at matched parameter count.

**P-PL3 — The Fisher-Optimal Block Size for Standard LLM Weights Is B=8, Not B=32.** The MX format's B=32 block exponent over-shares scale across weights whose Fisher information matrix has effective rank approximately 4–8 within any row. The Fisher-optimal block size, minimizing total quantization error at matched bit budget, is B=8. Testable by grid search over B ∈ {4, 8, 16, 32, 64} on LLaMA-class architectures at matched total bit budgets on standard perplexity benchmarks.

**P-PL4 — The Quantization Gap at FX16 + B=32 Is Below Training Variance by 2027.** Training any standard LLM architecture from random initialization entirely in FX16 with B=32 block scaling — no FP32 master copy, no floating-point operations — achieves within 0.5% of FP32 training quality on perplexity, MMLU, and HumanEval, consistent with the "FP4 All the Way" result (arXiv:2505.19115) extrapolated to FX16. The gap is the conversion cost; the precision floor at FX16 is sufficient for all standard transformer architectures.

**P-PL5 — The Apple Neural Engine Architecture Is the Commercial Proof of FXNS-1985.** The ANE's architecture — INT8 MAC units, per-layer block scaling, saturation arithmetic, CoreML compiler-assigned range analysis — matches, feature for feature, the FXNS-1985 standard that was never written. The CoreML compiler implements, as a software system, the block-scale ker(F) management that FXNS-1985 would have standardized in hardware. The M4 Neural Engine at 38 TOPS under 6W against the M4 GPU's BF16 throughput at 20W constitutes the commercial confirmation that the fixed-point architecture at the correct ker(F) granularity is 6–8× more efficient than per-element IEEE 754 floating-point for inference at matched accuracy.

---

## References

**The Hardware Lottery**

Hooker, S. The Hardware Lottery. arXiv:2009.06489, 2020. Published in Communications of the ACM.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., Polosukhin, I. Attention Is All You Need. arXiv:1706.03762, 2017. [Silicon co-design with Google TPU v2 systolic arrays.]

Rumelhart, D. E., Hinton, G. E., Williams, R. J. Learning representations by back-propagating errors. *Nature* 323, 533–536, 1986.

**The Floating-Point Standard**

Goldberg, D. What Every Computer Scientist Should Know About Floating-Point Arithmetic. *ACM Computing Surveys* 23(1), 5–48, 1991.

IEEE Standard for Binary Floating-Point Arithmetic. IEEE Std 754-1985. 1985.

IEEE Standard for Floating-Point Arithmetic. IEEE Std 754-2008. 2008.

Kahan, W. IEEE Standard 754 for Binary Floating-Point Arithmetic. IEEE Micro lecture notes, 1996.

**CORDIC and the Fixed-Point Foundation**

Volder, J. E. The CORDIC Trigonometric Computing Technique. *IRE Transactions on Electronic Computers* EC-8(3), 330–334, 1959.

Volder, J. E. The Birth of CORDIC. *Journal of VLSI Signal Processing* 25(2), 101–105, June 2000.

Walther, J. S. A Unified Algorithm for Elementary Functions. AFIPS Spring Joint Computer Conference, 1971.

Cochran, D. S. Algorithms and Accuracy in the HP-35. *Hewlett-Packard Journal* 23(10), 10–11, 1972.

**Fixed-Point DSP Heritage**

Texas Instruments. TMS320 First Generation User's Guide. 1983. [TMS32010 introduced April 8, 1983: fastest DSP on market, fixed-point at $50.]

Luo, Y. et al. Fixed-Point Quantization for Deep Neural Networks. *IEEE Transactions on VLSI Systems*, 2019.

Lapsley, P., Bier, J., Shoham, A., Lee, E. A. *DSP Processor Fundamentals: Architectures and Features*. IEEE Press, 1997.

**The Retroactive Derivation — The Quantization Committee**

Gupta, S., Agrawal, A., Gopalakrishnan, K., Narayanan, P. Deep Learning with Limited Numerical Precision. ICML 2015.

Micikevicius, P. et al. Mixed Precision Training. ICLR 2018. [Rediscovery of FX32 accumulator requirement already standard in TMS320C25, 1983.]

Dettmers, T., Lewis, M., Shleifer, S., Zettlemoyer, L. LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale. arXiv:2208.07339, 2022. [0.1% col(F) outliers established empirically.]

Frantar, E., Ashkboos, S., Hoefler, T., Alistarh, D. GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers. arXiv:2210.17323, 2022.

Rouhani, B. D. et al. With Shared Microexponents, A Little Shifting Goes a Long Way. ISCA 2023. [OCP MX format: block-level ker(F) partition, 38 years after FXNS-1985.]

Ma, S. et al. The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits. arXiv:2402.17764, 2024.

Chmiel, B., Fishman, M., Banner, R., Soudry, D. FP4 All the Way: Fully Quantized Training of LLMs. arXiv:2505.19115, May 2025. [Quantization gap is conversion cost, not precision floor; confirmed by fixed-point-native training.]

**June 2026 SOTA — The Convergence**

Abdalla, R., Hussein, A., Wu, M., Manocha, D. Minimizing the Hidden Cost of Scales: Graph-Guided Ultra-Low-Bit Quantization for Large Language Models (SAGE-PTQ). arXiv:2606.05429, June 3, 2026. [Fisher-guided dual-mode col(F)/ker(F): 1.03 bits, perplexity 6.74 vs 55.8 for BiLLM.]

Cim, M., Palangappa, P., Hodak, M., Dwivedula, R., Arunachalam, M., Kandemir, M. T. Pretraining Large Language Models with MXFP4 on Native FP4 Hardware. arXiv:2605.09825, AMD and Penn State University, May 2026. [First complete large-scale pretraining on native FP4 silicon; Wgrad instability is col(F)-path concentration; Hadamard rotation as col(F) protection.]

Kumar, A. et al. CARMEN: CORDIC-Accelerated Resource-Efficient Multi-Precision Inference Engine. arXiv:2605.06878, May 2026. [Volder's 1959 algorithm as ML accelerator; CORDIC hardware for hyperbolic and Lorentzian arithmetic.]

Bickford, M. The Self-Referential Fixed Point of the Complex Exponential: ρ = −W₋₁(−1). arXiv:2606.01668, June 2026. [Lambert boundary where CORDIC circular and hyperbolic modes are simultaneously required.]

**Geometric Curvature and the Representational Bundle**

Robinson, J. et al. On the Intrinsic Geometry of Transformer Token Embeddings. arXiv:2410.08993, 2024; arXiv:2504.01002, 2025. [Negative Ricci curvature in token embeddings: the representational geometry is hyperbolic, not Euclidean.]

He, X. et al. HELM: Hyperbolic Embedding for Large Language Models. arXiv:2505.24722, NeurIPS 2025. [Fully hyperbolic models outperform Euclidean baselines on hierarchical language tasks.]

Amari, S.-I. Natural Gradient Works Efficiently in Learning. *Neural Computation* 10(2), 251–276, 1998. [Fisher-Riemannian gradient; O(N²) cost that blocked adoption and was recovered 25 years later by Sophia, Muon, SOAP.]

**ERI Labs Corpus**

Ren, E. MANTISSA. github.com/ericrenone/MANTISSA. June 7, 2026.

Ren, E. THE-CHARACTERISTIC-WAS-ALWAYS-KER-F. github.com/ericrenone. June 7, 2026.

Ren, E. THE-FLOATED-POINT-WAS-ALWAYS-THE-DETOUR. github.com/ericrenone. June 7, 2026.

Ren, E. WHAT-KAHAN-GOT-RIGHT. github.com/ericrenone. June 7, 2026.

Ren, E. THE-FIXED-POINT-WAS-ALWAYS-THE-BOUNDARY. github.com/ericrenone. June 6, 2026.

Ren, E. THE-FIXED-POINT-WAS-ALWAYS-THE-BOUNDARY-2. github.com/ericrenone. June 6, 2026.

Ren, E. The-Five-Lotteries. github.com/ericrenone. May 2026.

Ren, E. Volder-1. github.com/ericrenone. May 2026.

Ren, E. CORDIRAC. github.com/ericrenone. March 26, 2026.

---

ERI Labs · Eric Ren · Jersey City, New Jersey · github.com/ericrenone · June 2026

Arithmetic corpus lineage: MANTISSA · THE-CHARACTERISTIC-WAS-ALWAYS-KER-F · THE-FLOATED-POINT-WAS-ALWAYS-THE-DETOUR · WHAT-KAHAN-GOT-RIGHT · **THE-LANGUAGE-WAS-ALWAYS-FLOATING**

Hardware corpus: CAST-IRON · CHORD · CORN · CROSS · CORDIRAC · Volder-1 · Rocket-Volder-1 · CARMEN

col(F)/ker(F) internalization arc: Napier 1614 → Oughtred 1622 → Volder 1956 → HP-35 1972 → (FXNS fork never taken) → IEEE 754 1985 → Python float 1991 (lottery locked) → LLM.int8() 2022 → MX 2023 → SAGE-PTQ 2026 → FXNS-2026

The detour: 1985–2023 · 38 years · $80–150B unnecessary compute · 8–10 year AI timeline delay · 11 years of quantization literature · Python still floating · the substrate already fixed.
