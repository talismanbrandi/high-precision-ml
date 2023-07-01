# High-precision regressors for particle physics
<h3>
<p>
    <a href="https://inspirehep.net/authors/1279838">Fady Bishara</a> (<a href="https://inspirehep.net/institutions/902770">DESY</a>), <a href="https://inspirehep.net/authors/1209756">Ayan Paul</a> (<a href="https://inspirehep.net/institutions/946076">Northeastern U. (main)</a>), <a href="https://inspirehep.net/authors/2628974">Jennifer Dy</a> (<a href="https://inspirehep.net/institutions/946076">Northeastern U. (main)</a>)</p>
  
  <p>
      e-Print:
          <a href="https://arxiv.org/abs/2302.00753">
      2302.00753
    </a>[physics.comp-ph]</p>
</h3>

## Data

The `data` folder contains three subfolders, one for each of the three phase-space dimensionalities we consider in the paper. The folders are organized as follows.

<pre><font color="#739FCF"><b>.</b></font>
├── <font color="#739FCF"><b>data</b></font>
│   ├── <font color="#739FCF"><b>2D</b></font>
│   │   ├── <font color="#739FCF"><b>test</b></font>
│   │   └── <font color="#739FCF"><b>train</b></font>
│   ├── <font color="#739FCF"><b>4D</b></font>
│   │   ├── <font color="#739FCF"><b>test</b></font>
│   │   └── <font color="#739FCF"><b>train</b></font>
│   └── <font color="#739FCF"><b>8D</b></font>
│       ├── <font color="#739FCF"><b>test</b></font>
│       └── <font color="#739FCF"><b>train</b></font>
</pre>

The `test` and `train` folders contain `CSV` file chunks with `D+2` columns were `D` is the phase-space dimension. The `D` coordinates are linearly mapped to the unit hypercube; they are labeled by $x_1,...,x_D$ in the header and span the first `D` columns of each file.

The `D+1`th column contains the un-normalized $f_{11}+f_{02}$ term while the `D+2`th column contains the normalized one, i.e., $(f_{11}+f_{02})/f_{00}$. Please refer to the paper for more details

## Code for training the regressors

The code used to train the BDTs and neural networks can be found in the `src` folder with the following structure:
<pre>└── <font color="#739FCF"><b>src</b></font>
    ├── <font color="#739FCF"><b>bdt</b></font>
    └── <font color="#739FCF"><b>dnn</b></font>
</pre>

For both the BDT and DNN cases, the folders contain a `python` script (either `bdt.py` or `dnn.py` as the case may be) which reads the settings from a config file (`config-in.json`). Each folder also contains a `README.md` file that explains the parameters in the config file.