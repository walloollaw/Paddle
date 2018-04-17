#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

################################################################################
#
# Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
This module provide a tool to facilitate the chainning of different readers
in model training or inference

more use case can be found in 'tests/test_pipeline.py'

Authors: wanglong03(wanglong03@baidu.com)
Date:    2018/04/03 14:27:06
"""

import json
import functools
import decorator


class PipelineError(ValueError):
    pass


def _batch(reader, batch_size):
    """
    Create a batched reader.

    :param reader: the data reader to read from.
    :type reader: callable
    :param batch_size: size of each mini-batch
    :type batch_size: int
    :return: the batched reader.
    :rtype: callable
    """

    def batch_reader():
        r = reader()
        b = []
        for instance in r:
            b.append(instance)
            if len(b) == batch_size:
                yield b
                b = []
        if b:
            yield b

    return batch_reader


def chain_funcs(funcs):
    """ chain a list of functions
    """

    def chained(*args, **kwargs):
        """ chained function
        """
        ret = funcs[0](*args, **kwargs)
        for f in funcs[1:]:
            ret = f(ret)
        return ret

    return chained


class Pipeline(object):
    """ a class to facilitate chainning the transformations applied to 'reader'
    """

    def __init__(self, reader):
        """ init
        """
        self._reader = reader
        self.reset()

    def reset(self, reader=None):
        """ reset the pipeline of transformations to initial state

        Args:
            reader (callable): a reader to provide data records

        Returns:
            None

        Raises:
            None
        """
        if reader is not None:
            self._reader = reader

        self._transformed = None
        self._pipeline = []

    def shuffle(self, size):
        """ shuffle the records in range of 'size'

        Args:
            size (int): size of shuffle range

        Returns:
            self

        Raises:
            None
        """
        self._pipeline.append(('shuffle', {'size': size}))
        return self

    def batch(self, size):
        """ make batches from data items

        Args:
            size (int): size of one batch

        Returns:
            self

        Raises:
            None
        """
        self._pipeline.append(('batch', {'size': size}))
        return self

    def map(self, f):
        """ do a function 'f' on every record in this reader

        Args:
            f (function): the function to be applied to every record

        Returns:
            self

        Raises:
            None
        """

        self._pipeline.append(('map', {'func': f}))
        return self

    def xmap(self, funcs, process_num, buffer_size, order=False):
        """ use multipleprocess to map samples from previouse reader

        Args:
            funcs (function or list): one map or a list of maps
            process_num (int): process number to handle original sample
            buffer_size (int): max buffer size
            order (bool): keep the order of the reader

        Returns:
            self

        Raises:
            PipelineError when param not valid
        """

        if type(funcs).__name__ == 'function':
            f = funcs
        elif type(funcs) is not []:
            f = chain_funcs(funcs)
        else:
            raise PipelineError(
                'invalid param for "funcs", not a function or a list of functions'
            )

        self._pipeline.append(('xmap', {'func': f, 'process_num': process_num, \
                'buffer_size': buffer_size, 'order': order}))

        return self

    def buffered(self, size):
        """ make the data records to be buffered without exceeding 'size' items

        Args:
            size (int): maximum buffer size

        Returns:
            self

        Raises:
            None
        """

        self._pipeline.append(('buffered', {'size': size}))
        return self

    def transform(self, reader):
        """ transform the 'reader' using transformations defined in self._pipeline

        Args:
            reader (callable): a reader to provide data records

        Returns:
            trans_reader (callable): a transformed reader

        Raises:
            PipelineError when not supported op_name appears
        """
        rd = reader
        for op_name, param in self._pipeline:
            if op_name == 'shuffle':
                rd = decorator.shuffle(rd, param['size'])
            elif op_name == 'batch':
                rd = _batch(rd, param['size'])
            elif op_name == 'buffered':
                rd = decorator.buffered(rd, param['size'])
            elif op_name == 'map':
                rd = decorator.map_readers(param['func'], rd)
            elif op_name == 'xmap':
                rd = decorator.xmap_readers(param['func'], rd, \
                        param['process_num'], param['buffer_size'], param['order'])
            else:
                raise PipelineError('not supported trasnfromation[%s]' %
                                    (op_name))

        return rd

    def reader(self):
        """ get the transformed reader from 'self._reader'

        Args:
            None

        Returns:
            reader (callable): transformed reader

        Raises:
            None
        """
        if self._transformed is None:
            self._transformed = self.transform(self._reader)

        return self._transformed

    def __str__(self):
        """ readable representation for this object, used to debug

        Args:
            None

        Returns:
            readable string for this object

        Raises:
            None
        """
        id = 0
        ops = ['\nPipeline:']
        if len(self._pipeline) == 0:
            return '\n  '.join(ops + ['empty'])

        for op_name, param in self._pipeline:
            ops.append("{id:%d, op:%s, param:%s}" % (id, op_name, str(param)))
            id += 1
        return '\n  '.join(ops)


if __name__ == "__main__":
    """ test
    """

    def data_reader():
        for i in xrange(10):
            yield i

    p = Pipeline(data_reader)
    print p
    rd = p.map(lambda r: 2 * r).shuffle(4).batch(2).reader()
    for i in rd():
        print i

#/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
