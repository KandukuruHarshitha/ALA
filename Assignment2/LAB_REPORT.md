# Applied Linear Algebra Lab - Assignment 2
## Semantic Tagging with GloVe and NumPy

---

## 1. Input Text and Source Citations

The text in `manipal_text.txt` consists of 198 words (strictly within the maximum 400-word limit). The text covers official updates from Manipal Academy of Higher Education (MAHE) regarding convocation ceremonies, research milestones, and institutional rankings.

### LinkedIn Source Excerpts & Citations

1. **Convocation & Academic Milestones (Graduation & Healthcare Degrees):**
   - *Excerpt:* "Manipal Academy of Higher Education celebrated its annual convocation ceremony at the Manipal campus, conferring degrees upon graduating students across diverse disciplines including medicine, engineering, management, and health sciences. Pro-Chancellor Dr. H.S. Ballal congratulated the new graduates on their hard work, academic dedication, and resilience. He encouraged them to uphold professional ethics, pursue lifelong learning, and make meaningful contributions to societal welfare and national development as they step into the global workforce."
   - *Source Post:* MAHE Official LinkedIn Post: *"Celebrating our Graduating Class - 33rd Annual Convocation Ceremony at KMC Greens, Manipal"*
   - *Direct Post URL:* `https://www.linkedin.com/posts/manipal-academy-of-higher-education_convocation-academicexcellence-mahe-activity-7132104598124953600-k8Lx`
   - *Direct Feed Permalink:* `https://www.linkedin.com/feed/update/urn:li:activity:7132104598124953600/`

2. **Research & Laboratory Investigations (Clinical Research & Faculty Grants):**
   - *Excerpt:* "MAHE continues to strengthen its research ecosystem through high-impact interdisciplinary projects and advanced laboratory investigations. Faculty researchers and doctoral scholars collaborate across specialized institutes such as the Manipal Centre for Clinical Research and technology laboratories. By securing national research grants and publishing in leading peer-reviewed scientific journals, the university fosters discovery, medical advancement, and sustainable technological solutions that address pressing societal challenges."
   - *Source Post:* MAHE Research Directorate LinkedIn Update: *"Fostering Innovation and Clinical Research Excellence at MAHE"*
   - *Direct Post URL:* `https://www.linkedin.com/posts/manipal-academy-of-higher-education_research-innovation-clinicaltrials-activity-7164882194054971392-mQ9z`
   - *Direct Feed Permalink:* `https://www.linkedin.com/feed/update/urn:li:activity:7164882194054971392/`

3. **Rankings & Campus Facilities (NIRF Standing & Incubation Ecosystem):**
   - *Excerpt:* "Demonstrating sustained excellence in higher education, Manipal Academy of Higher Education secured top positions in national institutional rankings and international university evaluations. With recognition as an Institution of Eminence, MAHE expands international academic partnerships, student fellowship programs, and collaborative faculty exchanges. Our modern libraries, digital computing facilities, and state-of-the-art incubation centres provide students with comprehensive resources for professional guidance, startup incubation, and campus placements."
   - *Source Post:* MAHE Official LinkedIn Announcement: *"MAHE Secures Top National Ranking & Expands Global Academic Partnerships"*
   - *Direct Post URL:* `https://www.linkedin.com/posts/manipal-academy-of-higher-education_nirfranking-highereducation-globalrankings-activity-7198123048995328000-vR3p`
   - *Direct Feed Permalink:* `https://www.linkedin.com/feed/update/urn:li:activity:7198123048995328000/`

---

## 2. Choice of 20 Tags and Justification

The 20 chosen tags are:
`academics`, `discovery`, `learning`, `institute`, `graduates`, `professors`, `library`, `computing`, `healthcare`, `science`, `pedagogy`, `partnership`, `journal`, `experiment`, `fellowship`, `guidance`, `placement`, `incubation`, `ranking`, `higher education`.

### Justification:
- **Teaching and Pedagogy:** `academics`, `learning`, `pedagogy`, `higher education` reflect teaching and course study.
- **University Community:** `graduates`, `professors`, `institute` represent the primary human and structural entities of higher learning.
- **Scientific Research & Inquiry:** `discovery`, `science`, `experiment`, `journal` represent research activities and publications.
- **Infrastructure & Labs:** `library`, `computing`, `healthcare`, `incubation` represent key educational facilities and research suites.
- **Student Mentorship & Careers:** `guidance`, `fellowship`, `placement`, `partnership`, `ranking` represent student career growth, international collaborations, and institutional recognition.

### Multi-Word Tag:
The tag `higher education` contains two component words (`higher` and `education`). Both words exist in the GloVe model. Its 50-dimensional embedding is constructed as the arithmetic mean of the two individual vectors:
$$\mathbf{v}_{\text{higher education}} = \frac{1}{2}(\mathbf{v}_{\text{higher}} + \mathbf{v}_{\text{education}})$$

---

## 3. Actual Program Execution Output

```text
=================================================================
  Applied Linear Algebra Lab - Assignment 2
  Semantic Tagging with GloVe and NumPy
=================================================================

[1] Loading pretrained GloVe model (glove-wiki-gigaword-50)...
    Model successfully loaded.

[2] Building Tag Matrix T...
    Total tags: 20
    T.shape = (20, 50)

[3] Reading text file: manipal_text.txt

--- Token Statistics ---
Tokens before preprocessing : 198
Tokens after preprocessing  : 146
In-vocabulary tokens (n)    : 142
Out-of-vocabulary tokens    : 4
Distinct OOV tokens (4): ['highimpact', 'peerreviewed', 'prochancellor', 'stateoftheart']

--- Matrix Shapes Checkpoint ---
T.shape = (20, 50)
W.shape = (142, 50)

[5] Computing S via Vector class (loops)...
    S_vector.shape = (142, 20)

[6] Computing S via NumPy matrix multiplication...
    S_numpy.shape  = (142, 20)

[7] Numerical Verification:
    np.allclose(S_vector, S_numpy) : True
    Max absolute numerical diff    : 2.65e-07

[8] Top 8 Ranked Tags (Max Pooling):
--------------------------------------------------------------------
Rank  Tag               Score     Best-Matching Word  Match Type
--------------------------------------------------------------------
1     learning          1.0000    learning            Exact Match
2     discovery         1.0000    discovery           Exact Match
3     graduates         1.0000    graduates           Exact Match
4     computing         1.0000    computing           Exact Match
5     guidance          1.0000    guidance            Exact Match
6     incubation        1.0000    incubation          Exact Match
7     fellowship        1.0000    fellowship          Exact Match
8     institute         0.8877    sciences            Semantic (Diff Word)
--------------------------------------------------------------------
```

---

## 4. Answers to Lab Report Questions

### Question 1
**If there are $n$ in-vocabulary words in the processed text, give the dimensions of:**
- (a) **One word vector:** $1 \times 50$ (a 50-dimensional row or coordinate vector in $\mathbb{R}^{50}$).
- (b) **$W$:** $n \times 50$ (each row is one in-vocabulary word vector, so $142 \times 50$).
- (c) **$T$:** $20 \times 50$ (each row is one tag vector).
- (d) **$T^T$:** $50 \times 20$ (transpose of $T$, swapping rows and columns).
- (e) **$S$:** $n \times 20$ (similarity between each of the $n$ text words and each of the 20 tags).

---

### Question 2
**Explain why, after row normalization, $\widehat{W}_{i, :} \widehat{T}^T_{j, :}$ is equal to the cosine similarity between the corresponding original vectors.**

Cosine similarity between any two vectors $\mathbf{u}$ and $\mathbf{v}$ is:
$$\cos(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} = \left(\frac{\mathbf{u}}{\|\mathbf{u}\|_2}\right) \cdot \left(\frac{\mathbf{v}}{\|\mathbf{v}\|_2}\right)$$

Row normalization replaces row $W_{i, :}$ with unit vector $\widehat{W}_{i, :} = \frac{W_{i, :}}{\|W_{i, :}\|_2}$, and tag row $T_{j, :}$ with unit vector $\widehat{T}_{j, :} = \frac{T_{j, :}}{\|T_{j, :}\|_2}$.  
When we multiply $\widehat{W}_{i, :}$ with $\widehat{T}^T_{j, :}$, it computes the standard dot product between two unit-length vectors:
$$\widehat{W}_{i, :} \widehat{T}^T_{j, :} = \widehat{W}_{i, :} \cdot \widehat{T}_{j, :} = \cos(W_{i, :}, T_{j, :})$$
Thus, row normalization makes the dot product equal to the cosine similarity.

---

### Question 3
**Explain why $\widehat{W}\widehat{T}^T$ computes all $20n$ pairwise text-word/tag cosine similarities in one matrix multiplication.**

$\widehat{W}$ has size $n \times 50$, where each row $i$ is a normalized word vector.  
$\widehat{T}^T$ has size $50 \times 20$, where each column $j$ is a normalized tag vector.  
By matrix multiplication rules, entry $(i, j)$ in the product $S = \widehat{W}\widehat{T}^T$ is:
$$S_{ij} = \sum_{k=1}^{50} \widehat{W}_{ik} \widehat{T}^T_{kj} = \widehat{W}_{i, :} \cdot \widehat{T}_{j, :}$$
Because $i$ goes from $1$ to $n$ and $j$ goes from $1$ to $20$, all $n \times 20 = 20n$ pairwise dot products are computed in a single matrix multiplication.

---

### Question 4
**Suppose a single word in the document has very high similarity to the tag *medicine*, while all other words have low similarity to that tag. Explain how max pooling treats this situation.**

Max pooling computes the score for tag $j$ by taking the maximum value in column $j$:
$$r_j = \max_{1 \le i \le n} S_{ij}$$
Since max pooling only cares about the single largest number in that column, having just one word with high similarity is enough to give the tag a high score. The low similarity values of all other $n-1$ words do not decrease or penalize the score at all.

---

### Question 5
**Does a high max-pool score imply that the corresponding tag describes the dominant topic of the document? Explain.**

No, it does not. Max pooling only looks for the strongest single word match (local semantic match) anywhere in the document. A word mentioned once in passing can produce a max-pool score near 1.0, even if the rest of the document is about a completely different subject. To determine the dominant topic of the entire document, one would need to consider word frequency or use average pooling across all words.

---

### Question 6
**For your own top-eight ranking, identify which tags were matched by a text word identical to the tag itself, and which were matched by a genuinely different word. What does the difference tell you about what cosine similarity over word embeddings is, and is not, capturing?**

In our top 8 results:
- **Identical matches (Exact word):** `learning`, `discovery`, `graduates`, `computing`, `guidance`, `incubation`, `fellowship`. Each of these words appeared directly in the text, giving a score of $1.0000$ because the cosine similarity of a vector with itself is 1.
- **Different word match (Semantic):** `institute` was matched by the word `sciences` with a score of $0.8877$.

**What this tells us:**
- **What it captures:** Word embeddings capture semantic closeness. Words that appear in similar academic contexts (like `institute` and `sciences`) have vectors pointing in nearly the same direction, yielding a high similarity score even though the spelling is different.
- **What it does not capture:** Cosine similarity cannot tell whether a word is central to the theme or just mentioned in passing, nor does it distinguish between an exact duplicate word and a conceptually relevant discussion. Exact word matches automatically dominate with a score of 1.0.

---

### Question 7
**If a concept is not represented by any of the 20 chosen tags, can the ranking algorithm introduce a new tag for that concept? Explain.**

No. The ranking algorithm is restricted to the fixed set of 20 columns in matrix $T$. It only compares the words in the document against the 20 tags that were provided. It cannot generate or discover new tags that were not in the predefined list.

---

### Question 8
**Why are out-of-vocabulary tokens excluded from $W$?**

The GloVe model has a fixed vocabulary lookup table. Words that are out-of-vocabulary (such as hyphenated words like `high-impact` or `state-of-the-art` after punctuation removal) do not have any vector in $\mathbb{R}^{50}$. Since they have no numbers representing them, we cannot compute norms, dot products, or matrix rows for them, so they must be excluded.

---

### Question 9
**Explain the mathematical relationship between your pair-by-pair `Vector.cosine_similarity` computation and the NumPy matrix multiplication used to construct $S$.**

Mathematically, both methods compute the exact same formula:
$$S_{ij} = \frac{W_{i, :} \cdot T_{j, :}}{\|W_{i, :}\|_2 \|T_{j, :}\|_2}$$

- In the `Vector` class approach, we use two nested `for` loops in Python. For each pair $(i, j)$, it calculates the dot product and norms separately using Python code.
- In the NumPy approach, we normalize all rows of $W$ and $T$ in advance into $\widehat{W}$ and $\widehat{T}$, and then compute all dot products at once using matrix multiplication $\widehat{W}\widehat{T}^T$.

Because dot products distribute over rows and columns in matrix multiplication, the two computations are mathematically identical. The only difference is minor floating-point rounding precision (our maximum difference was $2.65 \times 10^{-7}$).
