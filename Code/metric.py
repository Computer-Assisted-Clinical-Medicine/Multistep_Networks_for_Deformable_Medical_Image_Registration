import numpy as np
import pystrum.pynd.ndutils as nd

def dice_coefficient(output, target, threshold=0.5, smooth=1e-5):
    output = output[:, :, :] > threshold
    target = target[:, :, :] > threshold
    inse = np.count_nonzero(np.logical_and(output, target))
    l = np.count_nonzero(output)
    r = np.count_nonzero(target)
    hard_dice = (2 * inse + smooth) / (l + r + smooth)
    return hard_dice

def jc(disp):
    j_det=jacobian_determinant(disp)
    jc = (j_det <= 0).sum()
    return jc, j_det.mean()

def jc_proz(disp):
    jc_val, _ = jc(disp)
    ges = disp.shape[0] * disp.shape[1] * disp.shape[2]
    return jc_val / ges * 100

def jacobian_determinant(disp):
    """
    source: voxelmorph/voxelmorph/py/utils.py
    jacobian determinant of a displacement field.
    NB: to compute the spatial gradients, we use np.gradient.
    Parameters:
        disp: 2D or 3D displacement field of size [*vol_shape, nb_dims],
              where vol_shape is of len nb_dims
    Returns:
        jacobian determinant (scalar)
    """

    # check inputs
    volshape = disp.shape[:-1]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'

    # compute grid
    grid_lst = nd.volsize2ndgrid(volshape)
    grid = np.stack(grid_lst, len(volshape))

    # compute gradients
    J = np.gradient(disp + grid)

    # 3D glow
    if nb_dims == 3:
        dx = J[0]
        dy = J[1]
        dz = J[2]

        # compute jacobian components
        Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
        Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
        Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

        return Jdet0 - Jdet1 + Jdet2

    else:  # must be 2

        dfdx = J[0]
        dfdy = J[1]

        return dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]
