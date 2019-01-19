# Copyright (c) 2018 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

BOOST_PER_MOVE_NUMBER = (
    8.14308e-04, 1.56402e-02, 6.93261e-02, 1.32262e-01, 2.02702e-01, 2.77870e-01,
    3.54106e-01, 4.16647e-01, 4.70265e-01, 5.20241e-01, 5.67238e-01, 6.09612e-01,
    6.48552e-01, 6.85222e-01, 7.21541e-01, 7.52383e-01, 7.81914e-01, 8.07446e-01,
    8.29806e-01, 8.51154e-01, 8.70611e-01, 8.87777e-01, 9.03171e-01, 9.16842e-01,
    9.28338e-01, 9.38409e-01, 9.47081e-01, 9.55476e-01, 9.62205e-01, 9.67797e-01,
    9.72279e-01, 9.76051e-01, 9.79543e-01, 9.82489e-01, 9.84915e-01, 9.86856e-01,
    9.88480e-01, 9.90040e-01, 9.91350e-01, 9.92709e-01, 9.93771e-01, 9.94647e-01,
    9.95537e-01, 9.96032e-01, 9.96615e-01, 9.97059e-01, 9.97472e-01, 9.97781e-01,
    9.98108e-01, 9.98292e-01, 9.98443e-01, 9.98544e-01, 9.98645e-01, 9.98792e-01,
    9.98939e-01, 9.99068e-01, 9.99188e-01, 9.99252e-01, 9.99301e-01, 9.99377e-01,
    9.99430e-01, 9.99494e-01, 9.99547e-01, 9.99611e-01, 9.99664e-01, 9.99698e-01,
    9.99717e-01, 9.99729e-01, 9.99736e-01, 9.99751e-01, 9.99774e-01, 9.99770e-01,
    9.99777e-01, 9.99777e-01, 9.99784e-01, 9.99796e-01, 9.99795e-01, 9.99795e-01,
    9.99795e-01, 9.99810e-01, 9.99814e-01, 9.99809e-01, 9.99813e-01, 9.99817e-01,
    9.99813e-01, 9.99816e-01, 9.99820e-01, 9.99819e-01, 9.99823e-01, 9.99823e-01,
    9.99822e-01, 9.99834e-01, 9.99834e-01, 9.99833e-01, 9.99833e-01, 9.99833e-01,
    9.99832e-01, 9.99844e-01, 9.99848e-01, 9.99847e-01, 9.99847e-01, 9.99846e-01,
    9.99850e-01, 9.99850e-01, 9.99853e-01, 9.99849e-01, 9.99852e-01, 9.99852e-01,
    9.99860e-01, 9.99859e-01, 9.99859e-01, 9.99858e-01, 9.99858e-01, 9.99853e-01,
    9.99853e-01, 9.99852e-01, 9.99852e-01, 9.99851e-01, 9.99850e-01, 9.99850e-01,
    9.99849e-01, 9.99849e-01, 9.99848e-01, 9.99847e-01, 9.99847e-01, 9.99846e-01,
    9.99845e-01, 9.99849e-01, 9.99853e-01, 9.99852e-01, 9.99856e-01, 9.99859e-01,
    9.99859e-01, 9.99858e-01, 9.99852e-01, 9.99856e-01, 9.99865e-01, 9.99869e-01,
    9.99868e-01, 9.99867e-01, 9.99866e-01, 9.99865e-01, 9.99864e-01, 9.99863e-01,
    9.99862e-01, 9.99861e-01, 9.99860e-01, 9.99859e-01, 9.99858e-01, 9.99867e-01,
    9.99877e-01, 9.99876e-01, 9.99875e-01, 9.99874e-01, 9.99873e-01, 9.99871e-01,
    9.99870e-01, 9.99869e-01, 9.99868e-01, 9.99867e-01, 9.99866e-01, 9.99864e-01,
    9.99869e-01, 9.99867e-01, 9.99866e-01, 9.99865e-01, 9.99869e-01, 9.99868e-01,
    9.99867e-01, 9.99865e-01, 9.99864e-01, 9.99862e-01, 9.99867e-01, 9.99866e-01,
    9.99864e-01, 9.99856e-01, 9.99861e-01, 9.99860e-01, 9.99858e-01, 9.99856e-01,
    9.99855e-01, 9.99860e-01, 9.99865e-01, 9.99864e-01, 9.99862e-01, 9.99860e-01,
    9.99866e-01, 9.99864e-01, 9.99863e-01, 9.99861e-01, 9.99859e-01, 9.99865e-01,
    9.99863e-01, 9.99853e-01, 9.99851e-01, 9.99849e-01, 9.99856e-01, 9.99863e-01,
    9.99861e-01, 9.99859e-01, 9.99857e-01, 9.99855e-01, 9.99853e-01, 9.99851e-01,
    9.99849e-01, 9.99847e-01, 9.99845e-01, 9.99842e-01, 9.99840e-01, 9.99838e-01,
    9.99836e-01, 9.99833e-01, 9.99831e-01, 9.99828e-01, 9.99837e-01, 9.99835e-01,
    9.99832e-01, 9.99830e-01, 9.99827e-01, 9.99825e-01, 9.99834e-01, 9.99831e-01,
    9.99829e-01, 9.99826e-01, 9.99861e-01, 9.99859e-01, 9.99857e-01, 9.99855e-01,
    9.99853e-01, 9.99850e-01, 9.99848e-01, 9.99846e-01, 9.99843e-01, 9.99841e-01,
    9.99839e-01, 9.99836e-01, 9.99849e-01, 9.99846e-01, 9.99844e-01, 9.99841e-01,
    9.99839e-01, 9.99836e-01, 9.99833e-01, 9.99831e-01, 9.99828e-01, 9.99825e-01,
    9.99822e-01, 9.99819e-01, 9.99835e-01, 9.99832e-01, 9.99829e-01, 9.99826e-01,
    9.99823e-01, 9.99820e-01, 9.99817e-01, 9.99896e-01, 9.99895e-01, 9.99893e-01,
    9.99891e-01, 9.99889e-01, 9.99887e-01, 9.99885e-01, 9.99883e-01, 9.99881e-01,
    9.99878e-01, 9.99876e-01, 9.99899e-01, 9.99897e-01, 9.99895e-01, 9.99893e-01,
    9.99891e-01, 9.99889e-01, 9.99887e-01, 9.99913e-01, 9.99912e-01, 9.99910e-01,
    9.99908e-01, 9.99906e-01, 9.99936e-01, 9.99935e-01, 9.99934e-01, 9.99932e-01,
    9.99931e-01, 9.99930e-01, 9.99964e-01, 9.99964e-01, 9.99963e-01, 9.99962e-01,
    9.99961e-01, 9.99961e-01, 9.99960e-01, 9.99959e-01, 9.99958e-01, 9.99958e-01,
    9.99957e-01, 9.99956e-01, 9.99955e-01, 9.99954e-01, 9.99954e-01, 9.99953e-01,
    9.99952e-01, 9.99951e-01, 9.99950e-01, 9.99949e-01, 9.99948e-01, 9.99947e-01,
    9.99946e-01, 9.99946e-01, 9.99945e-01, 9.99944e-01, 9.99943e-01, 9.99942e-01,
    9.99941e-01, 9.99940e-01, 9.99939e-01, 9.99938e-01, 9.99937e-01, 9.99936e-01,
    9.99935e-01, 9.99934e-01, 9.99933e-01, 9.99932e-01, 9.99931e-01, 9.99929e-01,
    9.99928e-01, 9.99927e-01, 9.99926e-01, 9.99925e-01, 9.99924e-01, 9.99923e-01
)
