"""
Theano op for multiclass hinge loss courtesy James Bergstra.
    http://trac-hg.assembla.com/pylearn/changeset/
        d4a35c1c0a232fdec05df19387438d9c1638e06c
"""

# This file seems like it has some overlap with theano.tensor.nnet.  Which functions should go
# in which file?

from theano import gof
from theano.gof import Apply
from theano import tensor
from theano.tensor import DisconnectedType
import numpy as np

class MultiHingeMargin(gof.Op):
    """
    This is a hinge loss function for multiclass predictions.

    For each vector X[i] and label index yidx[i],
    output z[i] = 1 - margin

    where margin is the difference between X[i, yidx[i]] and the maximum other element of X[i].
    """
    default_output = 0
    def __eq__(self, other):
        return type(self) == type(other)
    def __hash__(self):
        return tensor.hashtype(self)
    def __str__(self):
        return self.__class__.__name__
    def make_node(self, X, yidx):
        X_ = tensor.as_tensor_variable(X)
        yidx_ = tensor.as_tensor_variable(yidx)
        if X_.type.ndim != 2:
            raise TypeError('X must be matrix')
        if yidx.type.ndim != 1:
            raise TypeError('yidx must be vector')
        if 'int' not in str(yidx.type.dtype):
            raise TypeError("yidx must be integers, it's a vector of class labels")
        hinge_loss = tensor.vector(dtype=X.dtype)
        winners = X.type()
        return Apply(self, [X_, yidx_], [hinge_loss, winners])
    def perform(self, node, input_storage, out):
        X, yidx = input_storage
        toplabel = X.shape[1]-1
        out[0][0] = z = np.zeros_like(X[:,0])
        out[1][0] = w = np.zeros_like(X)
        for i,Xi in enumerate(X):
            yi = yidx[i]
            if yi == 0:
                next_best = Xi[1:].argmax()+1
            elif yi==toplabel:
                next_best = Xi[:toplabel].argmax()
            else:
                next_best0 = Xi[:yi].argmax()
                next_best1 = Xi[yi+1:].argmax()+yi+1
                next_best = next_best0 if Xi[next_best0]>Xi[next_best1] else next_best1
            margin = Xi[yi] - Xi[next_best]
            if margin < 1:
                z[i] = 1 - margin
                w[i,yi] = -1
                w[i,next_best] = 1
    def grad(self, inputs, g_outs):
        z = self(*inputs)
        w = z.owner.outputs[1]
        gz, gw = g_outs
        gX = gz.dimshuffle(0,'x') * w
        if gw is None:
            gY = None
        elif isinstance(gw.type, DisconnectedType):
            gY = DisconnectedType()()
        else:
            raise NotImplementedError()
        return [gX, gY]
    # def c_code_cache_version(self):
    #     return (1,)
    # def c_code(self, node, name, (X, y_idx), (z,w), sub):
    #     return '''
    #     if ((%(X)s->descr->type_num != PyArray_DOUBLE) && (%(X)s->descr->type_num != PyArray_FLOAT))
    #     {
    #         PyErr_SetString(PyExc_TypeError, "types should be float or float64");
    #         %(fail)s;
    #     }
    #     if ((%(y_idx)s->descr->type_num != PyArray_INT64)
    #         && (%(y_idx)s->descr->type_num != PyArray_INT32)
    #         && (%(y_idx)s->descr->type_num != PyArray_INT16)
    #         && (%(y_idx)s->descr->type_num != PyArray_INT8))
    #     {
    #         PyErr_SetString(PyExc_TypeError, "y_idx not int8, int16, int32, or int64");
    #         %(fail)s;
    #     }
    #     if ((%(X)s->nd != 2)
    #         || (%(y_idx)s->nd != 1))
    #     {
    #         PyErr_SetString(PyExc_ValueError, "rank error");
    #         %(fail)s;
    #     }
    #     if (%(X)s->dimensions[0] != %(y_idx)s->dimensions[0])
    #     {
    #         PyErr_SetString(PyExc_ValueError, "dy.shape[0] != sm.shape[0]");
    #         %(fail)s;
    #     }
    #     if ((NULL == %(z)s)
    #         || (%(z)s->dimensions[0] != %(X)s->dimensions[0]))
    #     {
    #         Py_XDECREF(%(z)s);
    #         %(z)s = (PyArrayObject*) PyArray_SimpleNew(1, PyArray_DIMS(%(X)s),
    #                                                     type_num_%(X)s);
    #         if (!%(z)s)
    #         {
    #             PyErr_SetString(PyExc_MemoryError, "failed to alloc dx output");
    #             %(fail)s;
    #         }
    #     }
    #     if ((NULL == %(w)s)
    #         || (%(w)s->dimensions[0] != %(X)s->dimensions[0])
    #         || (%(w)s->dimensions[1] != %(X)s->dimensions[1]))
    #     {
    #         Py_XDECREF(%(w)s);
    #         %(w)s = (PyArrayObject*) PyArray_SimpleNew(2, PyArray_DIMS(%(X)s),
    #                                                     type_num_%(X)s);
    #         if (!%(w)s)
    #         {
    #             PyErr_SetString(PyExc_MemoryError, "failed to alloc dx output");
    #             %(fail)s;
    #         }
    #     }

    #     for (size_t i = 0; i < %(X)s->dimensions[0]; ++i)
    #     {
    #         const dtype_%(X)s* __restrict__ X_i = (dtype_%(X)s*) (%(X)s->data + %(X)s->strides[0] * i);
    #         npy_intp SX = %(X)s->strides[1]/sizeof(dtype_%(X)s);

    #         dtype_%(w)s* __restrict__ w_i = (dtype_%(w)s*) (%(w)s->data + %(w)s->strides[0] * i);
    #         npy_intp Sw = %(w)s->strides[1]/sizeof(dtype_%(w)s);

    #         const dtype_%(y_idx)s y_i = ((dtype_%(y_idx)s*)(%(y_idx)s->data + %(y_idx)s->strides[0] * i))[0];

    #         dtype_%(X)s X_i_max = X_i[0];
    #         dtype_%(X)s X_at_y_i = X_i[0];
    #         size_t X_i_argmax = 0;
    #         size_t j = 1;
    #         w_i[0] = 0;

    #         if (y_i == 0)
    #         {
    #             X_i_max = X_i[SX];
    #             X_i_argmax = 1;
    #             w_i[Sw] = 0;
    #         }
    #         for (; j < %(X)s->dimensions[1]; ++j)
    #         {
    #             dtype_%(X)s  X_ij = X_i[j*SX];
    #             if (j == y_i)
    #             {
    #                 X_at_y_i = X_ij;
    #             }
    #             else if (X_ij > X_i_max)
    #             {
    #                 X_i_max = X_ij;
    #                 X_i_argmax = j;
    #             }
    #             w_i[j*Sw] = 0;
    #         }
    #         if (0 < 1 - X_at_y_i + X_i_max)
    #         {
    #             ((dtype_%(z)s*)(%(z)s->data + %(z)s->strides[0] * i))[0]
    #                 = 1 - X_at_y_i + X_i_max;
    #             w_i[y_i*Sw] = -1;
    #             w_i[X_i_argmax*Sw] = 1;
    #         }
    #     }
    #     ''' % dict(locals(), **sub)

multi_hinge_margin = MultiHingeMargin()
